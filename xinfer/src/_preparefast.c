/* PREPARE-C PC1/PC2: GIL-releasing prepare hot-path pack for xinfer (wallx2b).
 *
 * The HOST-GAP decomposition (R/host-gap, 2026-11-15) showed the prepare window's
 * python work serializes with the view-pool workers and the engine device-image
 * thread on one GIL (~2.6 ms phase): the main thread alone holds the GIL ~0.63 ms
 * per request, of which pin/state/mrope/dict/pad are the whole story once K1-K6
 * landed. These helpers move the deterministic integer/float64 tail of
 * Wallx2bInputProcessor.__call__ into C calls whose compute loops run under
 * Py_BEGIN_ALLOW_THREADS:
 *
 *   state_text(agent_pos, agent_mask, norm_min, norm_delta, bins, clamp)
 *       -> (text str, values int64[n], n_values, sum_abs_digits) | None
 *       _normalize_state + np.digitize(right=False == searchsorted 'right') +
 *       masked select + " ".join(str(v)) -- float64/int64 arithmetic exactly as
 *       numpy computed it, same digits as str().
 *
 *   tail(token_ids, image_token_id, merge_size, grids, delay, rtc_gap, width,
 *        rtc_actions, p, h)
 *       -> dict of numpy arrays | None
 *       mrope_position_ids (K4 math) + action_positions (+rtc_rope_gap on rows
 *       >= delay) + action_mask (RTC pattern) + pad_prompt (K3 math: zero id
 *       padding, ONE position padding, truncated prompt region) +
 *       image_token_index + prompt_valid + rtc_actions zero-padded -- the fresh
 *       arrays the python path builds, in one pass.
 *
 * Anything non-canonical (dtype, shape, mismatched image runs) is declined with
 * None and the caller runs the original python path, so accepted results are
 * equal by construction; the action gate then sees at most a schedule change.
 * Pure CPython + Numpy C API (built like _wirefast; no torch headers).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pthread.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_VERSION_BUILD
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* A C-contiguous float64 array of any shape, or NULL (declined) without error. */
static PyArrayObject *as_f64_c(PyObject *arg) {
    if (arg == NULL || !PyArray_Check(arg)) return NULL;
    PyArrayObject *arr = (PyArrayObject *)arg;
    if (PyArray_TYPE(arr) != NPY_DOUBLE || !PyArray_IS_C_CONTIGUOUS(arr)) return NULL;
    return arr;
}

static PyArrayObject *as_i64_c(PyObject *arg) {
    if (arg == NULL || !PyArray_Check(arg)) return NULL;
    PyArrayObject *arr = (PyArrayObject *)arg;
    if (PyArray_TYPE(arr) != NPY_INT64 || !PyArray_IS_C_CONTIGUOUS(arr)) return NULL;
    return arr;
}

/* ------------------------------------------------------------------ */
/* state_text                                                         */
/* ------------------------------------------------------------------ */

static PyObject *pf_state_text(PyObject *self, PyObject *args) {
    PyObject *pos_o, *mask_o, *min_o, *delta_o, *bins_o;
    int clamp;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOOp", &pos_o, &mask_o, &min_o, &delta_o, &bins_o, &clamp))
        return NULL;
    PyArrayObject *pos = as_f64_c(pos_o), *mask = as_f64_c(mask_o);
    PyArrayObject *min = as_f64_c(min_o), *delta = as_f64_c(delta_o), *bins = as_f64_c(bins_o);
    if (!pos || !mask || !min || !delta || !bins) Py_RETURN_NONE;
    npy_intp n = PyArray_SIZE(pos);
    if (PyArray_SIZE(mask) != n || PyArray_SIZE(min) != n || PyArray_SIZE(delta) != n)
        Py_RETURN_NONE;
    npy_intp n_bins = PyArray_SIZE(bins);
    const double *xp = (const double *)PyArray_DATA(pos);
    const double *mp = (const double *)PyArray_DATA(mask);
    const double *mn = (const double *)PyArray_DATA(min);
    const double *dv = (const double *)PyArray_DATA(delta);
    const double *bp = (const double *)PyArray_DATA(bins);

    PyObject *values = PyArray_SimpleNew(1, &n, NPY_INT64);
    if (values == NULL) return NULL;
    int64_t *out = (int64_t *)PyArray_DATA((PyArrayObject *)values);
    npy_intp count = 0;
    int canonical = 1;
    Py_ssize_t text_len = 0;   /* join length: sum(len(str(v))) + (count-1) spaces */
    Py_ssize_t abs_digits = 0; /* sum(len(str(abs(v)))) for text_length_range */
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < n; i++) {
        if (mp[i] == 0.0) continue;
        /* ((x - min) / delta) * 2 - 1 in float64, exactly numpy's op order */
        double v = ((xp[i] - mn[i]) / dv[i]) * 2.0 - 1.0;
        if (isnan(v)) { canonical = 0; break; }  /* numpy's NaN bin belongs to the fallback */
        if (clamp) {
            if (v > 1.0) v = 1.0;
            if (v < -1.0) v = -1.0;
        }
        /* np.digitize(v, bins, right=False) == searchsorted(bins, v, 'right')
         * == count of bins <= v; the python code subtracts one. */
        npy_intp le = 0;
        for (npy_intp b = 0; b < n_bins; b++) if (bp[b] <= v) le++;
        int64_t digit = (int64_t)le - 1;
        out[count] = digit;
        uint64_t m = digit < 0 ? (uint64_t)(-(digit + 1)) + 1u : (uint64_t)digit;
        int nd = 1;
        while (m >= 10u) { m /= 10u; nd++; }
        text_len += nd + (digit < 0 ? 1 : 0);
        abs_digits += nd;
        count++;
    }
    Py_END_ALLOW_THREADS
    if (!canonical) { Py_DECREF(values); Py_RETURN_NONE; }
    if (count > 0) text_len += count - 1;
    {
        PyArray_Dims newshape = {&count, 1};
        PyObject *resized = PyArray_Resize((PyArrayObject *)values, &newshape, 0, NPY_ANYORDER);
        if (resized == NULL) {
            Py_DECREF(values);
            return NULL;
        }
        Py_DECREF(resized);
        out = (int64_t *)PyArray_DATA((PyArrayObject *)values);
    }
    PyObject *text;
    if (count == 0) {
        text = PyUnicode_FromString("");
    } else {
        char *buf = (char *)PyMem_RawMalloc((size_t)text_len + 1u);
        if (buf == NULL) { Py_DECREF(values); return PyErr_NoMemory(); }
        Py_ssize_t w = 0;
        Py_BEGIN_ALLOW_THREADS
        for (npy_intp i = 0; i < count; i++) {
            if (i > 0) buf[w++] = ' ';
            int64_t v = out[i];
            w += sprintf(buf + w, "%lld", (long long)v);
        }
        Py_END_ALLOW_THREADS
        text = PyUnicode_DecodeASCII(buf, (Py_ssize_t)strlen(buf), NULL);
        PyMem_RawFree(buf);
    }
    if (text == NULL) { Py_DECREF(values); return NULL; }
    PyObject *result = PyTuple_New(4);
    if (result == NULL) { Py_DECREF(text); Py_DECREF(values); return NULL; }
    PyTuple_SET_ITEM(result, 0, text);                                   /* steals */
    PyTuple_SET_ITEM(result, 1, values);                                 /* steals */
    PyObject *n_obj = PyLong_FromSsize_t((Py_ssize_t)count);
    PyObject *d_obj = PyLong_FromSsize_t(abs_digits);
    if (n_obj == NULL || d_obj == NULL) {
        Py_XDECREF(n_obj); Py_XDECREF(d_obj); Py_DECREF(result);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 2, n_obj);
    PyTuple_SET_ITEM(result, 3, d_obj);
    return result;
}

/* ------------------------------------------------------------------ */
/* handoff (PC4): the view-pool worker's wire chain in one C call      */
/* ------------------------------------------------------------------ */

static unsigned char pf_b64_value[256];

static void pf_b64_init(void) {
    memset(pf_b64_value, 0xFF, sizeof(pf_b64_value));
    for (int i = 0; i < 26; i++) pf_b64_value['A' + i] = (unsigned char)i;
    for (int i = 0; i < 26; i++) pf_b64_value['a' + i] = (unsigned char)(26 + i);
    for (int i = 0; i < 10; i++) pf_b64_value['0' + i] = (unsigned char)(52 + i);
    pf_b64_value['+'] = 62;
    pf_b64_value['/'] = 63;
}

/* ---------------- HOST-FINAL HF2: chunk-parallel large-string decode ----------------
 * The decode is position-pure (quad q only reads src[4q..4q+3] and writes
 * dst[3q..3q+2]), so splitting the quads into contiguous ranges produces
 * byte-identical output, and each range applies the same first-touch quad
 * validation; any range failing declines the whole call exactly as the serial
 * loop would -- the accept/decline set is unchanged. The 3 static workers are
 * created once; the calling thread decodes chunk 0. At most one parallel
 * decode runs at a time (a second large handoff falls back to the serial
 * path), and everything runs inside the existing Py_BEGIN_ALLOW_THREADS
 * section, on idle cores BEFORE the wave's tickets are out (the device is
 * idle then; the PC5 lesson about competing with the DVPP host supply chain
 * does not apply pre-wave). */
#define PF_B64_PARALLEL_MIN 131072
#define PF_B64_CHUNKS 4

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    int live;      /* 0 = not started, 1 = up, -1 = gave up */
    int busy;      /* a parallel decode owns the pool */
    int pending;   /* bitmask of dispatched worker chunks still running */
    int ok_mask;   /* per-chunk results (bit set = chunk failed) */
    const unsigned char *src;
    unsigned char *dst;
    Py_ssize_t q0[PF_B64_CHUNKS - 1], q1[PF_B64_CHUNKS - 1];
    Py_ssize_t quads, pad;
} pf_b64_pool_t;

static pf_b64_pool_t pf_b64_pool = {
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER,
    0, 0, 0, 0, NULL, NULL, {0, 0, 0}, {0, 0, 0}, 0, 0,
};

static int pf_b64_range(const unsigned char *src, unsigned char *dst,
                        Py_ssize_t q0, Py_ssize_t q1, Py_ssize_t quads, Py_ssize_t pad) {
    /* returns 0 on success, 1 on the first invalid quad (same math and order
     * as the serial loop; only the quad numbered quads-1 takes pad branches) */
    for (Py_ssize_t q = q0; q < q1; q++) {
        const unsigned char *p = src + 4 * q;
        unsigned v0 = pf_b64_value[p[0]], v1 = pf_b64_value[p[1]];
        unsigned v2 = pf_b64_value[p[2]], v3 = pf_b64_value[p[3]];
        unsigned char *d = dst + 3 * q;
        if (v0 == 0xFF || v1 == 0xFF) return 1;
        if (q + 1 == quads) {
            if (pad < 2 && v2 == 0xFF) return 1;
            if (pad < 1 && v3 == 0xFF) return 1;
            unsigned trip = (v0 << 18) | (v1 << 12);
            if (pad < 2) trip |= v2 << 6;
            if (pad == 0) trip |= v3;
            d[0] = (unsigned char)(trip >> 16);
            if (pad != 2) d[1] = (unsigned char)(trip >> 8);
            if (pad == 0) d[2] = (unsigned char)trip;
        } else {
            if (v2 == 0xFF || v3 == 0xFF) return 1;
            unsigned trip = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
            d[0] = (unsigned char)(trip >> 16);
            d[1] = (unsigned char)(trip >> 8);
            d[2] = (unsigned char)trip;
        }
    }
    return 0;
}

static void *pf_b64_worker(void *arg) {
    long id = (long)arg;
    pf_b64_pool_t *pool = &pf_b64_pool;
    pthread_mutex_lock(&pool->mu);
    for (;;) {
        int bit = 1 << id;
        while (!(pool->busy && (pool->pending & bit)))
            pthread_cond_wait(&pool->cv_work, &pool->mu);
        const unsigned char *src = pool->src;
        unsigned char *dst = pool->dst;
        Py_ssize_t q0 = pool->q0[id], q1 = pool->q1[id];
        Py_ssize_t quads = pool->quads, pad = pool->pad;
        pthread_mutex_unlock(&pool->mu);
        int failed = pf_b64_range(src, dst, q0, q1, quads, pad);
        pthread_mutex_lock(&pool->mu);
        if (failed) pool->ok_mask |= bit;
        /* HOST-FINAL fix: the pending bit marks the chunk IN FLIGHT and is
         * cleared only here, after the worker stopped touching dst -- the
         * caller returns (and may free the bytes object on decline) the
         * moment pending hits zero, so clearing at dispatch raced the free. */
        pool->pending &= ~bit;
        pthread_cond_broadcast(&pool->cv_done);
    }
    return NULL;
}

static int pf_b64_pool_start(void) {
    pf_b64_pool_t *pool = &pf_b64_pool;
    pthread_mutex_lock(&pool->mu);
    if (pool->live == 0) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        int ok = 1;
        for (long i = 0; i < PF_B64_CHUNKS - 1; i++) {
            pthread_t tid;
            if (pthread_create(&tid, &attr, pf_b64_worker, (void *)i) != 0) { ok = 0; break; }
        }
        pthread_attr_destroy(&attr);
        pool->live = ok ? 1 : -1;
    }
    int live = pool->live == 1;
    pthread_mutex_unlock(&pool->mu);
    return live;
}

/* returns 1 when it decoded (or declined) the whole string, 0 = caller must
 * take the serial path */
static int pf_b64_parallel(const unsigned char *src, unsigned char *dst,
                           Py_ssize_t quads, Py_ssize_t pad, Py_ssize_t n) {
    if (n < PF_B64_PARALLEL_MIN || quads < PF_B64_CHUNKS) return 0;
    pf_b64_pool_t *pool = &pf_b64_pool;
    if (!pf_b64_pool_start()) return 0;
    Py_ssize_t per = quads / PF_B64_CHUNKS;
    Py_ssize_t bounds[PF_B64_CHUNKS + 1];
    for (int c = 0; c < PF_B64_CHUNKS; c++) bounds[c] = per * c;
    bounds[PF_B64_CHUNKS] = quads;  /* the last quad (pad rules) goes to chunk PF_B64_CHUNKS-1 */
    pthread_mutex_lock(&pool->mu);
    if (pool->busy) {  /* another large decode owns the pool: serial for us */
        pthread_mutex_unlock(&pool->mu);
        return 0;
    }
    pool->src = src;
    pool->dst = dst;
    pool->quads = quads;
    pool->pad = pad;
    pool->ok_mask = 0;
    pool->pending = 0;
    for (int c = 1; c < PF_B64_CHUNKS; c++) {
        pool->q0[c - 1] = bounds[c];
        pool->q1[c - 1] = bounds[c + 1];
        if (pool->q1[c - 1] > pool->q0[c - 1]) pool->pending |= 1 << (c - 1);
    }
    pool->busy = 1;
    pthread_cond_broadcast(&pool->cv_work);
    pthread_mutex_unlock(&pool->mu);
    /* chunk 0 runs on the calling thread */
    int failed = pf_b64_range(src, dst, bounds[0], bounds[1], quads, pad);
    pthread_mutex_lock(&pool->mu);
    while (pool->pending)
        pthread_cond_wait(&pool->cv_done, &pool->mu);
    if (failed) pool->ok_mask |= 1;
    int bad = pool->ok_mask != 0;
    pool->busy = 0;
    pthread_cond_broadcast(&pool->cv_work);
    pthread_mutex_unlock(&pool->mu);
    return bad ? -1 : 1;
}

/* canonical base64 -> bytes (None = declined), the _wirefast.b64decode math */
static PyObject *pf_b64(PyObject *arg) {
    Py_ssize_t n = 0;
    const char *s = PyUnicode_AsUTF8AndSize(arg, &n);
    if (s == NULL) { PyErr_Clear(); return NULL; }
    Py_ssize_t pad = 0;
    if (n % 4 != 0) return NULL;
    if (n >= 1 && s[n - 1] == '=') pad++;
    if (n >= 2 && s[n - 2] == '=') pad++;
    Py_ssize_t quads = n / 4;
    Py_ssize_t out_len = 3 * quads - pad;
    if (out_len < 0) out_len = 0;
    PyObject *out = PyBytes_FromStringAndSize(NULL, out_len);
    if (out == NULL) return NULL;
    if (out_len > 0) {
        unsigned char *dst = (unsigned char *)PyBytes_AS_STRING(out);
        const unsigned char *src = (const unsigned char *)s;
        int ok = 1;
        Py_BEGIN_ALLOW_THREADS
        int prc = pf_b64_parallel(src, dst, quads, pad, n);
        if (prc != 0) {
            ok = prc > 0;
        } else for (Py_ssize_t q = 0; q < quads; q++) {
            const unsigned char *p = src + 4 * q;
            unsigned v0 = pf_b64_value[p[0]], v1 = pf_b64_value[p[1]];
            unsigned v2 = pf_b64_value[p[2]], v3 = pf_b64_value[p[3]];
            if (v0 == 0xFF || v1 == 0xFF) { ok = 0; break; }
            if (q + 1 == quads) {
                /* data positions [0, 4-pad) must be valid; the '=' suffix
                 * positions are '=' by the pad counting above and are ignored
                 * exactly as the old [0, n-pad) validation ignored them */
                if (pad < 2 && v2 == 0xFF) { ok = 0; break; }
                if (pad < 1 && v3 == 0xFF) { ok = 0; break; }
                unsigned trip = (v0 << 18) | (v1 << 12);
                if (pad < 2) trip |= v2 << 6;
                if (pad == 0) trip |= v3;
                dst[0] = (unsigned char)(trip >> 16);
                if (pad != 2) dst[1] = (unsigned char)(trip >> 8);
                if (pad == 0) dst[2] = (unsigned char)trip;
            } else {
                if (v2 == 0xFF || v3 == 0xFF) { ok = 0; break; }
                unsigned trip = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
                dst[0] = (unsigned char)(trip >> 16);
                dst[1] = (unsigned char)(trip >> 8);
                dst[2] = (unsigned char)trip;
            }
            dst += 3;
        }
        Py_END_ALLOW_THREADS
        if (!ok) { Py_DECREF(out); return NULL; }
    }
    return out;
}

/* python jpeg_size: SOI + marker walk to the SOF frame header; -1 = decline */
static int pf_jpeg_size(const unsigned char *data, Py_ssize_t n, long long *height, long long *width) {
    if (n < 2 || data[0] != 0xFF || data[1] != 0xD8) return -1;
    Py_ssize_t i = 2;
    while (i + 9 <= n) {
        if (data[i] != 0xFF) return -1;
        unsigned char marker = data[i + 1];
        if (marker == 0xFF) { i += 1; continue; }
        if (marker == 0x01 || (0xD0 <= marker && marker <= 0xD8)) { i += 2; continue; }
        if (0xC0 <= marker && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC) {
            *height = (data[i + 5] << 8) | data[i + 6];
            *width = (data[i + 7] << 8) | data[i + 8];
            return 0;
        }
        i += 2 + ((data[i + 2] << 8) | data[i + 3]);
    }
    return -1;
}

/* python smart_resize: round() is half-to-even (rint under FE_TONEAREST),
 * sqrt is exactly-rounded per IEEE-754 like numpy's, floor/ceil exact. */
static int pf_smart_resize(long long height, long long width, long long factor,
                           long long min_pixels, long long max_pixels,
                           long long *out_h, long long *out_w) {
    if (factor <= 0) return -1;
    double hh = (double)height, ww = (double)width;
    double ratio = (height > width ? (double)height / (double)width : (double)width / (double)height);
    if (ratio > 200.0) return -1;  /* the python path raises; decline instead */
    long long h_bar = (long long)llrint(hh / (double)factor) * factor;
    long long w_bar = (long long)llrint(ww / (double)factor) * factor;
    if (h_bar < factor) h_bar = factor;
    if (w_bar < factor) w_bar = factor;
    if ((double)h_bar * (double)w_bar > (double)max_pixels) {
        double beta = sqrt((double)height * (double)width / (double)max_pixels);
        h_bar = (long long)floor(hh / beta / (double)factor) * factor;
        w_bar = (long long)floor(ww / beta / (double)factor) * factor;
    } else if ((double)h_bar * (double)w_bar < (double)min_pixels) {
        double beta = sqrt((double)min_pixels / ((double)height * (double)width));
        h_bar = (long long)ceil(hh * beta / (double)factor) * factor;
        w_bar = (long long)ceil(ww * beta / (double)factor) * factor;
    }
    *out_h = h_bar;
    *out_w = w_bar;
    return 0;
}

/* handoff(value, target, resize_factor, resize_min_pixels, resize_max_pixels,
 *         grid_factor, min_pixels, max_pixels)
 *   -> (data, (h0, w0), ((h1, w1), (h2, w2))) | None
 * The K6 worker chain before device_image: _wire_view_fast (b64), is_jpeg +
 * jpeg_size header scan, resize_sizes' two smart_resize calls. Base64 str or
 * raw bytes only (the benchmark and production wire form); arrays and
 * non-canonical encodings decline to the python chain unchanged.
 */
static PyObject *pf_handoff(PyObject *self, PyObject *args) {
    PyObject *value;
    long long target, rf, rmin, rmax, gf, minp, maxp;
    (void)self;
    if (!PyArg_ParseTuple(args, "Onllllll", &value, &target, &rf, &rmin, &rmax, &gf, &minp, &maxp))
        return NULL;
    PyObject *data = NULL;
    if (PyUnicode_Check(value)) {
        data = pf_b64(value);
        if (data == NULL) Py_RETURN_NONE;
    } else if (PyBytes_Check(value)) {
        data = value;
        Py_INCREF(data);
    } else if (PyByteArray_Check(value) || PyObject_CheckBuffer(value)) {
        Py_buffer view;
        if (PyObject_GetBuffer(value, &view, PyBUF_SIMPLE) != 0) { PyErr_Clear(); Py_RETURN_NONE; }
        data = PyBytes_FromStringAndSize((const char *)view.buf, view.len);
        PyBuffer_Release(&view);
        if (data == NULL) return NULL;
    } else {
        Py_RETURN_NONE;
    }
    long long h0 = 0, w0 = 0;
    const unsigned char *bytes = (const unsigned char *)PyBytes_AS_STRING(data);
    Py_ssize_t n = PyBytes_GET_SIZE(data);
    if (pf_jpeg_size(bytes, n, &h0, &w0) != 0) {
        Py_DECREF(data);
        Py_RETURN_NONE;
    }
    /* resize_sizes: first pass with the resize_factor bounds, second with the
     * patch grid bounds, exactly as the python two-smart_resize chain. */
    long long h1 = 0, w1 = 0, h2 = 0, w2 = 0;
    if (target == -1) {
        h1 = h0; w1 = w0;
    } else if (w0 > h0) {
        w1 = target;
        h1 = (long long)((double)target * (double)h0 / (double)w0);
    } else {
        h1 = target;
        w1 = (long long)((double)target * (double)w0 / (double)h0);
    }
    if (pf_smart_resize(h1, w1, rf, rmin, rmax, &h1, &w1) != 0) {
        Py_DECREF(data);
        Py_RETURN_NONE;
    }
    if (pf_smart_resize(h1, w1, gf, minp, maxp, &h2, &w2) != 0) {
        Py_DECREF(data);
        Py_RETURN_NONE;
    }
    PyObject *sizes2 = Py_BuildValue("((ll)(ll))", h1, w1, h2, w2);
    PyObject *src = Py_BuildValue("(ll)", h0, w0);
    PyObject *out = PyTuple_New(3);
    if (out == NULL || sizes2 == NULL || src == NULL) {
        Py_XDECREF(out); Py_XDECREF(sizes2); Py_XDECREF(src);
        Py_DECREF(data);
        return NULL;
    }
    PyTuple_SET_ITEM(out, 0, data);
    PyTuple_SET_ITEM(out, 1, src);
    PyTuple_SET_ITEM(out, 2, sizes2);
    return out;
}

/* ------------------------------------------------------------------ */
/* tail                                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    npy_intp start, end;   /* image token run [start, end) */
    npy_intp rows, cols;
    int64_t block_advance;
} grid_run;

static PyObject *pf_tail(PyObject *self, PyObject *args) {
    PyObject *ids_o, *grids_o, *rtc_o;
    Py_ssize_t image_token_id, merge_size, delay, rtc_gap, width, p, h;
    (void)self;
    if (!PyArg_ParseTuple(args, "OnnOnnnOnn", &ids_o, &image_token_id, &merge_size, &grids_o,
                          &delay, &rtc_gap, &width, &rtc_o, &p, &h))
        return NULL;
    PyArrayObject *ids = as_i64_c(ids_o);
    if (!ids || PyArray_NDIM(ids) > 2) Py_RETURN_NONE;
    npy_intp L = PyArray_SIZE(ids);
    if (merge_size <= 0 || width <= 0 || delay < 0 || delay > h || p <= 0 || h <= 0 ||
        p + h != L || p != (Py_ssize_t)(npy_intp)p)
        Py_RETURN_NONE;
    const int64_t *tk = (const int64_t *)PyArray_DATA(ids);

    /* grids: sequence of (t, gh, gw); rows = gh // merge, cols = gw // merge */
    PyObject *seq = PySequence_Fast(grids_o, "grids must be a sequence");
    if (seq == NULL) { PyErr_Clear(); Py_RETURN_NONE; }
    Py_ssize_t n_grids = PySequence_Fast_GET_SIZE(seq);
    grid_run runs[16];
    if (n_grids < 0 || n_grids > 16) { Py_DECREF(seq); Py_RETURN_NONE; }
    for (Py_ssize_t g = 0; g < n_grids; g++) {
        PyObject *tup = PySequence_Tuple(PySequence_Fast_GET_ITEM(seq, g));
        if (tup == NULL || PyTuple_GET_SIZE(tup) != 3) {
            Py_XDECREF(tup); Py_DECREF(seq); PyErr_Clear(); Py_RETURN_NONE;
        }
        long long gh = PyLong_AsLongLong(PyTuple_GET_ITEM(tup, 1));
        long long gw = PyLong_AsLongLong(PyTuple_GET_ITEM(tup, 2));
        Py_DECREF(tup);
        if (PyErr_Occurred()) { PyErr_Clear(); Py_DECREF(seq); Py_RETURN_NONE; }
        runs[g].rows = (npy_intp)(gh / merge_size);
        runs[g].cols = (npy_intp)(gw / merge_size);
        if (runs[g].rows <= 0 || runs[g].cols <= 0) { Py_DECREF(seq); Py_RETURN_NONE; }
    }
    Py_DECREF(seq);

    /* pass 1 (GIL released): locate+validate the image runs, build the position
     * walk exactly as K4's advance/cumsum does, count prompt image tokens. */
    int64_t *position = (int64_t *)PyMem_RawMalloc((size_t)L * sizeof(int64_t));
    if (position == NULL) Py_RETURN_NONE;
    npy_intp img_count = 0;
    int ok = 1;
    Py_BEGIN_ALLOW_THREADS
    npy_intp cursor = 0;
    for (Py_ssize_t g = 0; ok && g < n_grids; g++) {
        npy_intp count = runs[g].rows * runs[g].cols;
        npy_intp start = -1;
        for (npy_intp i = cursor; i < (npy_intp)L; i++) {
            if (tk[i] == (int64_t)image_token_id) { start = i; break; }
        }
        if (start < 0) { ok = 0; break; }
        npy_intp end = start + count;
        if (end > (npy_intp)L) { ok = 0; break; }
        for (npy_intp i = start; i < end; i++) {
            if (tk[i] != (int64_t)image_token_id) { ok = 0; break; }
        }
        if (!ok) break;
        runs[g].start = start;
        runs[g].end = end;
        runs[g].block_advance = runs[g].rows > runs[g].cols ? runs[g].rows : runs[g].cols;
        cursor = end;
    }
    if (ok) {
        for (npy_intp i = cursor; i < (npy_intp)L; i++) {
            if (tk[i] == (int64_t)image_token_id) { ok = 0; break; }
        }
    }
    if (ok) {
        /* position[0] = 0; position[i] = position[i-1] + advance[i-1], where a
         * text token advances 1, an image token 0 and each run's LAST token
         * additionally carries its block advance (K4: advance[end-1] += max). */
        int64_t pos_running = 0;
        for (npy_intp i = 0; i < (npy_intp)L; i++) {
            position[i] = pos_running;
            int64_t adv = (tk[i] == (int64_t)image_token_id) ? 0 : 1;
            for (Py_ssize_t g = 0; g < n_grids; g++) {
                if (runs[g].end - 1 == i) adv += runs[g].block_advance;
            }
            pos_running += adv;
        }
        for (npy_intp i = 0; i < p; i++) {
            if (tk[i] == (int64_t)image_token_id) img_count++;
        }
    }
    Py_END_ALLOW_THREADS
    if (!ok) { PyMem_RawFree(position); Py_RETURN_NONE; }

    /* optional rtc: float32 [d, D] C-contiguous with d == delay */
    PyArrayObject *rtc = NULL;
    npy_intp rtc_D = 0;
    if (rtc_o != Py_None) {
        if (!PyArray_Check(rtc_o) || PyArray_TYPE((PyArrayObject *)rtc_o) != NPY_FLOAT32 ||
            !PyArray_IS_C_CONTIGUOUS((PyArrayObject *)rtc_o) ||
            PyArray_NDIM((PyArrayObject *)rtc_o) != 2 ||
            PyArray_DIM((PyArrayObject *)rtc_o, 0) != (npy_intp)delay) {
            PyMem_RawFree(position);
            Py_RETURN_NONE;
        }
        rtc = (PyArrayObject *)rtc_o;
        rtc_D = PyArray_DIM(rtc, 1);
    }

    /* allocate the result arrays (GIL held) */
    npy_intp keep = width < p ? width : p;
    npy_intp pad_n = width > keep ? width - keep : 0;
    npy_intp dims2[2] = {1, width};
    npy_intp dims3[3] = {3, 1, width};
    npy_intp dimh3[3] = {3, 1, h};
    npy_intp dimh[4] = {1, 1, h, width + h};
    npy_intp dim1[1] = {1};
    PyArrayObject *ids_out = (PyArrayObject *)PyArray_SimpleNew(2, dims2, NPY_INT64);
    PyArrayObject *pos_out = (PyArrayObject *)PyArray_SimpleNew(3, dims3, NPY_INT64);
    PyArrayObject *apos_out = (PyArrayObject *)PyArray_SimpleNew(3, dimh3, NPY_INT64);
    PyArrayObject *amask_out = (PyArrayObject *)PyArray_SimpleNew(4, dimh, NPY_BOOL);
    PyArrayObject *pvalid_out = (PyArrayObject *)PyArray_SimpleNew(1, dim1, NPY_INT64);
    PyArrayObject *imgidx_out = (PyArrayObject *)PyArray_SimpleNew(1, &img_count, NPY_INT64);
    PyArrayObject *rtc_out = NULL;
    if (rtc != NULL) {
        npy_intp dimr[3] = {1, h, rtc_D};
        rtc_out = (PyArrayObject *)PyArray_SimpleNew(3, dimr, NPY_FLOAT32);
    }
    if (!ids_out || !pos_out || !apos_out || !amask_out || !pvalid_out || !imgidx_out ||
        (rtc != NULL && !rtc_out)) {
        Py_XDECREF(ids_out); Py_XDECREF(pos_out); Py_XDECREF(apos_out);
        Py_XDECREF(amask_out); Py_XDECREF(pvalid_out); Py_XDECREF(imgidx_out);
        Py_XDECREF(rtc_out);
        PyMem_RawFree(position);
        return NULL;
    }

    /* pass 2 (GIL released): fill everything */
    {
        int64_t *ids_dst = (int64_t *)PyArray_DATA(ids_out);
        int64_t *pos_dst = (int64_t *)PyArray_DATA(pos_out);
        int64_t *apos_dst = (int64_t *)PyArray_DATA(apos_out);
        char *amask_dst = (char *)PyArray_DATA(amask_out);
        int64_t *pvalid_dst = (int64_t *)PyArray_DATA(pvalid_out);
        int64_t *imgidx_dst = (int64_t *)PyArray_DATA(imgidx_out);
        Py_BEGIN_ALLOW_THREADS
        /* input_ids: first `keep` prompt tokens then zero padding (K3:
         * concatenate(ids[:, :width], zeros); width < p truncates to keep). */
        for (npy_intp c = 0; c < keep; c++) ids_dst[c] = tk[c];
        for (npy_intp c = keep; c < width; c++) ids_dst[c] = 0;
        /* position_ids: prompt positions then ONE padding per axis (K3). */
        for (int a = 0; a < 3; a++) {
            int64_t *row = pos_dst + (npy_intp)a * width;
            for (npy_intp c = 0; c < keep; c++) row[c] = position[c];
            for (npy_intp c = keep; c < width; c++) row[c] = 1;
        }
        /* image blocks overwrite their run (K4: out[0]=block, out[1]=block+r,
         * out[2]=block+c); cols >= keep are dropped as the python slice did. */
        for (Py_ssize_t g = 0; g < n_grids; g++) {
            npy_intp start = runs[g].start, end = runs[g].end;
            if (start >= p) continue;
            int64_t block = position[start];
            npy_intp r = 0, c = 0;
            for (npy_intp i = start; i < end; i++) {
                if (c == runs[g].cols) { c = 0; r++; }
                if (i < keep) {
                    pos_dst[i] = block;
                    pos_dst[width + i] = block + (int64_t)r;
                    pos_dst[2 * width + i] = block + (int64_t)c;
                }
                c++;
            }
        }
        /* action_positions: positions[p:] (+rtc_gap on rows >= delay). */
        for (int a = 0; a < 3; a++) {
            int64_t *row = apos_dst + (npy_intp)a * h;
            for (npy_intp j = 0; j < h; j++) {
                row[j] = position[p + j] + (j >= delay ? (int64_t)rtc_gap : 0);
            }
        }
        /* action_mask: ones; zero padding spans [keep, keep+pad_n); action cols
         * map to original col p + j; rows < delay mask original cols >= p+delay. */
        for (npy_intp r = 0; r < h; r++) {
            char *row = amask_dst + r * (width + h);
            for (npy_intp c = 0; c < width + h; c++) {
                int v = 1;
                if (c >= keep && c < keep + pad_n) {
                    v = 0;
                } else if (c >= keep + pad_n) {
                    npy_intp acol = p + (c - keep - pad_n);
                    if (r < delay && acol >= p + delay) v = 0;
                }
                row[c] = (char)v;
            }
        }
        pvalid_dst[0] = p < width ? p : width;
        npy_intp w = 0;
        for (npy_intp i = 0; i < p; i++) {
            if (tk[i] == (int64_t)image_token_id) imgidx_dst[w++] = i;
        }
        if (rtc_out != NULL) {
            float *dst = (float *)PyArray_DATA(rtc_out);
            memset(dst, 0, sizeof(float) * (size_t)(h * rtc_D));
            if (delay > 0) {
                memcpy(dst, PyArray_DATA(rtc), sizeof(float) * (size_t)(delay * rtc_D));
            }
        }
        Py_END_ALLOW_THREADS
    }
    PyMem_RawFree(position);

    PyObject *out = PyDict_New();
    int bad = out == NULL ||
        PyDict_SetItemString(out, "input_ids", (PyObject *)ids_out) ||
        PyDict_SetItemString(out, "position_ids", (PyObject *)pos_out) ||
        PyDict_SetItemString(out, "action_positions", (PyObject *)apos_out) ||
        PyDict_SetItemString(out, "action_mask", (PyObject *)amask_out) ||
        PyDict_SetItemString(out, "prompt_valid", (PyObject *)pvalid_out) ||
        PyDict_SetItemString(out, "image_token_index", (PyObject *)imgidx_out) ||
        (rtc_out != NULL && PyDict_SetItemString(out, "rtc_actions", (PyObject *)rtc_out));
    Py_DECREF(ids_out); Py_DECREF(pos_out); Py_DECREF(apos_out);
    Py_DECREF(amask_out); Py_DECREF(pvalid_out); Py_DECREF(imgidx_out);
    Py_XDECREF(rtc_out);
    if (bad) { Py_XDECREF(out); return NULL; }
    return out;
}

static PyMethodDef pf_methods[] = {
    {"state_text", pf_state_text, METH_VARARGS,
     "state_text(agent_pos, agent_mask, norm_min, norm_delta, bins, clamp) -> "
     "(text, values int64[n], n_values, sum_abs_digits) | None."},
    {"tail", pf_tail, METH_VARARGS,
     "tail(token_ids, image_token_id, merge_size, grids, delay, rtc_gap, width, rtc_actions, p, h) "
     "-> dict of numpy arrays | None."},
    {"handoff", pf_handoff, METH_VARARGS,
     "handoff(value, target, resize_factor, resize_min_pixels, resize_max_pixels, grid_factor, "
     "min_pixels, max_pixels) -> (data, (h0, w0), ((h1, w1), (h2, w2))) | None."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef pf_module = {
    PyModuleDef_HEAD_INIT,
    "_preparefast",
    "GIL-releasing prepare tail pack (canonical wallx2b shapes only; None = python fallback).",
    -1,
    pf_methods,
};

PyMODINIT_FUNC PyInit__preparefast(void) {
    import_array();
    pf_b64_init();
    return PyModule_Create(&pf_module);
}

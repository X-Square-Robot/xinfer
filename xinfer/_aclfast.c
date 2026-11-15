/* _aclfast.c -- HOST-TAIL HT1: GIL-releasing wrappers for the hot DVPP/acl
 * host calls of the engine's device-image thread.
 *
 * The compiled CANN python binding (acl.so) holds the interpreter lock for
 * the whole duration of every himpi/rt call; the start chain of one view
 * (dvpp_get_image_info + 2x dvpp_malloc + aclrtMemcpy + vdec_send_stream)
 * and the wave's fetch chain (vdec_get_frame + aclrtMemcpyAsync +
 * vdec_release_frame) therefore serialize the GIL phase for ~0.4-0.8 ms per
 * request.  This extension resolves the same native symbols from the already
 * loaded CANN libraries and calls them under Py_BEGIN_ALLOW_THREADS, in the
 * same order, with the same struct layouts the binding builds, from the same
 * (device-image) thread -- so the channel/stream state machine is unchanged;
 * only the lock is released while the driver runs.
 *
 * Struct layouts were recovered from the binding's own converters
 * (acl.so @ cann-9.1.1; disassembly of the dvpp_get_image_info /
 * vdec_send_stream / vdec_get_frame / vdec_release_frame / dvpp_malloc /
 * dvpp_free wrappers -- host-tail REPORT section 3).  Sizes are asserted
 * against the bytes the binding zeroes/memsets.
 *
 * Every function declines (returns None) when the natives could not be
 * resolved, so npu_images falls back to the binding call path unchanged.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ---- native struct layouts (byte-exact, from the binding's converters) -- */

typedef struct {              /* hi_vdec_stream; binding zeroes 48 bytes */
    uint32_t end_of_frame;    /* +0x00 */
    uint32_t end_of_stream;   /* +0x04 */
    uint32_t need_display;    /* +0x08 */
    uint32_t pad0;            /* +0x0c */
    uint64_t pts;             /* +0x10 */
    uint64_t private_data;    /* +0x18 */
    uint32_t len;             /* +0x20 */
    uint32_t pad1;            /* +0x24 */
    uint64_t addr;            /* +0x28 */
} ht_vdec_stream;             /* 48 */

typedef struct {              /* vdec_send_stream's picture arg; zeroes 48 */
    uint32_t width;           /* +0x00 */
    uint32_t height;          /* +0x04 */
    uint32_t width_stride;    /* +0x08 */
    uint32_t height_stride;   /* +0x0c */
    uint32_t pixel_format;    /* +0x10 */
    uint32_t pad;             /* +0x14 */
    uint64_t vir_addr;        /* +0x18 */
    uint32_t buffer_size;     /* +0x20 */
    uint8_t rsv[0x30 - 0x24]; /* crop offsets etc, zeroed, unused */
} ht_vdec_picture;            /* 48 */

typedef struct {              /* dvpp_get_image_info out; zeroes 40 */
    uint32_t width;           /* +0x00 */
    uint32_t height;          /* +0x04 */
    uint32_t width_stride;    /* +0x08 */
    uint32_t height_stride;   /* +0x0c */
    uint32_t img_buf_size;    /* +0x10 */
    uint32_t pixel_format;    /* +0x14 */
    uint32_t reserved[4];     /* +0x18 */
} ht_image_info;              /* 40 */

typedef struct {              /* hi_video_frame; binding memsets 0x148 */
    uint32_t width, height, field, pixel_format, video_format, compress_mode;
    uint32_t dynamic_range, color_gamut;              /* +0x00..0x1c */
    uint32_t header_stride[3];                        /* +0x20 */
    uint32_t width_stride[3];                         /* +0x2c */
    uint32_t height_stride[3];                        /* +0x38 */
    uint32_t pad0;                                    /* +0x44 */
    uint64_t header_phys[3];                          /* +0x48 */
    uint64_t phys_addr[3];                            /* +0x60 */
    uint64_t header_virt[3];                          /* +0x78 */
    uint64_t virt_addr[3];                            /* +0x90 */
    uint32_t time_ref;                                /* +0xa8 */
    uint32_t pad1;                                    /* +0xac */
    uint64_t pts;                                     /* +0xb0 */
    uint64_t user_data[2];                            /* +0xb8 */
    uint32_t frame_flag;                              /* +0xc8 */
    uint8_t rsv[0x148 - 0xcc];                        /* +0xcc..0x147 */
} ht_video_frame;              /* 328 */

_Static_assert(sizeof(ht_vdec_stream) == 48, "stream layout");
_Static_assert(sizeof(ht_vdec_picture) == 48, "picture layout");
_Static_assert(sizeof(ht_image_info) == 40, "info layout");
_Static_assert(sizeof(ht_video_frame) == 328, "frame layout");
_Static_assert(offsetof(ht_video_frame, virt_addr) == 0x90, "virt_addr");
_Static_assert(offsetof(ht_video_frame, frame_flag) == 0xc8, "frame_flag");
_Static_assert(offsetof(ht_vdec_stream, addr) == 0x28, "stream addr");
_Static_assert(offsetof(ht_vdec_picture, vir_addr) == 0x18, "picture vir_addr");

/* ---- resolved natives (RTLD_NOLOAD first: acl.so already loaded them) --- */

typedef int (*f_get_image_info)(uint32_t, ht_vdec_stream *, ht_image_info *);
typedef int (*f_dvpp_malloc)(int32_t, void **, uint64_t);
typedef int (*f_dvpp_free)(void *);
typedef int (*f_send)(uint32_t, ht_vdec_stream *, ht_vdec_picture *, uint32_t);
typedef int (*f_get_frame)(uint32_t, ht_video_frame *, void *, void *, uint32_t);
typedef int (*f_release)(uint32_t, ht_video_frame *);
typedef int (*f_memcpy)(void *, size_t, const void *, size_t, int);
typedef int (*f_memcpy_async)(void *, size_t, const void *, size_t, int, void *);

static f_get_image_info n_get_info;
static f_dvpp_malloc n_malloc;
static f_dvpp_free n_free;
static f_send n_send;
static f_get_frame n_get_frame;
static f_release n_release;
static f_memcpy n_memcpy;
static f_memcpy_async n_memcpy_async;

static void *ht_dlopen(const char *soname) {
    void *h = dlopen(soname, RTLD_NOW | RTLD_NOLOAD);
    if (h == NULL) h = dlopen(soname, RTLD_NOW);
    return h;
}

static int resolve_all(void) {
    void *mpi = ht_dlopen("libacl_dvpp_mpi.so");
    void *cl = ht_dlopen("libascendcl.so");
    if (mpi == NULL || cl == NULL) return 0;
#define HT_SYM(h, name) dlsym(h, #name)
    n_get_info = (f_get_image_info)HT_SYM(mpi, hi_mpi_dvpp_get_image_info);
    n_malloc = (f_dvpp_malloc)HT_SYM(mpi, hi_mpi_dvpp_malloc);
    n_free = (f_dvpp_free)HT_SYM(mpi, hi_mpi_dvpp_free);
    n_send = (f_send)HT_SYM(mpi, hi_mpi_vdec_send_stream);
    n_get_frame = (f_get_frame)HT_SYM(mpi, hi_mpi_vdec_get_frame);
    n_release = (f_release)HT_SYM(mpi, hi_mpi_vdec_release_frame);
    n_memcpy = (f_memcpy)HT_SYM(cl, aclrtMemcpy);
    n_memcpy_async = (f_memcpy_async)HT_SYM(cl, aclrtMemcpyAsync);
#undef HT_SYM
    return n_get_info && n_malloc && n_free && n_send && n_get_frame && n_release
        && n_memcpy && n_memcpy_async;
}

static PyObject *af_init(PyObject *self, PyObject *noargs) {
    (void)self; (void)noargs;
    static int done = 0, ok = 0;
    if (!done) { ok = resolve_all(); done = 1; }
    if (!ok) Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

/* probe(payload_type, addr, size) ->
 *   (status, width, height, width_stride, height_stride, img_buf_size,
 *    pixel_format)  [None when unresolved] */
static PyObject *af_probe(PyObject *self, PyObject *args) {
    (void)self;
    unsigned int payload; unsigned long long addr, size;
    if (!PyArg_ParseTuple(args, "IKK", &payload, &addr, &size)) return NULL;
    if (n_get_info == NULL) Py_RETURN_NONE;
    ht_vdec_stream stream;
    ht_image_info info;
    memset(&stream, 0, sizeof stream);
    memset(&info, 0, sizeof info);
    stream.end_of_frame = 1; stream.need_display = 1; stream.len = (uint32_t)size;
    stream.addr = addr;
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = n_get_info(payload, &stream, &info);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("iIIIIII", status, info.width, info.height,
                         info.width_stride, info.height_stride,
                         info.img_buf_size, info.pixel_format);
}

/* start(dev, chn, jpeg_addr, size, out_size, width, height, wstride,
 *       hstride, pixel_format, timeout_ms) -> (status, phase, source, output)
 * phases: 0 src-malloc 1 h2d-copy 2 out-malloc 3 send (source/output are
 * valid only for phase 3; earlier failures free what was allocated, exactly
 * as _launch's except blocks do).  A send failure (phase 3, status != 0)
 * hands both pointers back: the caller frees them and drops the channel. */
static PyObject *af_start(PyObject *self, PyObject *args) {
    (void)self;
    int dev, chn, pixel_format, timeout;
    unsigned long long jpeg_addr, size, out_size, width, height, wstride, hstride;
    if (!PyArg_ParseTuple(args, "iiKKKKIIIIi", &dev, &chn, &jpeg_addr, &size,
                          &out_size, &width, &height, &wstride, &hstride,
                          &pixel_format, &timeout)) return NULL;
    if (n_malloc == NULL || n_send == NULL || n_memcpy == NULL) Py_RETURN_NONE;

    ht_vdec_stream stream;
    ht_vdec_picture picture;
    memset(&stream, 0, sizeof stream);
    memset(&picture, 0, sizeof picture);
    stream.end_of_frame = 1; stream.need_display = 1;
    stream.len = (uint32_t)size;
    picture.width = (uint32_t)width; picture.height = (uint32_t)height;
    picture.width_stride = (uint32_t)wstride; picture.height_stride = (uint32_t)hstride;
    picture.pixel_format = (uint32_t)pixel_format; picture.buffer_size = (uint32_t)out_size;

    void *source = NULL, *output = NULL;
    int status = 0, phase = 3;
    Py_BEGIN_ALLOW_THREADS
    status = n_malloc(dev, &source, size);
    if (status != 0) { phase = 0; goto out; }
    status = n_memcpy(source, (size_t)size, (const void *)(uintptr_t)jpeg_addr,
                      (size_t)size, 1 /*host->device*/);
    if (status != 0) { phase = 1; n_free(source); source = NULL; goto out; }
    stream.addr = (uint64_t)(uintptr_t)source;
    status = n_malloc(dev, &output, out_size);
    if (status != 0) { phase = 2; n_free(source); source = NULL; goto out; }
    picture.vir_addr = (uint64_t)(uintptr_t)output;
    status = n_send((uint32_t)chn, &stream, &picture, (uint32_t)timeout);
    if (status != 0) phase = 3;  /* caller frees both and drops the channel */
out: ;
    Py_END_ALLOW_THREADS
    return Py_BuildValue("iiKK", status, phase,
                         (unsigned long long)(uintptr_t)source,
                         (unsigned long long)(uintptr_t)output);
}

/* start3(dev, (chn, jpeg_addr, size, out_size, width, height, wstride,
 *         hstride, pixel_format), ...x N..., timeout_ms) ->
 *   [(status, phase, source, output)] * N
 * WAVE-TAIL WT2c: the whole ticket batch's start chains run under ONE GIL
 * release, so the prepare threads' python slices cannot interleave between
 * the three [malloc + h2d + malloc + send] sequences.  Each item's phases
 * and cleanup are af_start's, item for item; the chains run in list order,
 * the same order the per-ticket calls would have issued them. */
static PyObject *af_start3(PyObject *self, PyObject *args) {
    (void)self;
    int dev, timeout;
    int chn0, chn1, chn2, fmt0, fmt1, fmt2;
    unsigned long long ja0, sz0, out0, w0, h0, ws0, hs0;
    unsigned long long ja1, sz1, out1, w1, h1, ws1, hs1;
    unsigned long long ja2, sz2, out2, w2, h2, ws2, hs2;
    if (!PyArg_ParseTuple(args, "i(iKKKKIIII)(iKKKKIIII)(iKKKKIIII)i",
                          &dev,
                          &chn0, &ja0, &sz0, &out0, &w0, &h0, &ws0, &hs0, &fmt0,
                          &chn1, &ja1, &sz1, &out1, &w1, &h1, &ws1, &hs1, &fmt1,
                          &chn2, &ja2, &sz2, &out2, &w2, &h2, &ws2, &hs2, &fmt2,
                          &timeout)) return NULL;
    if (n_malloc == NULL || n_send == NULL || n_memcpy == NULL) Py_RETURN_NONE;

    struct { int chn; unsigned long long ja, sz, out, w, h, ws, hs; int fmt; } spec[3]
        = {{chn0, ja0, sz0, out0, w0, h0, ws0, hs0, fmt0},
           {chn1, ja1, sz1, out1, w1, h1, ws1, hs1, fmt1},
           {chn2, ja2, sz2, out2, w2, h2, ws2, hs2, fmt2}};
    void *source[3] = {NULL, NULL, NULL};
    void *output[3] = {NULL, NULL, NULL};
    int status[3] = {0, 0, 0};
    int phase[3] = {3, 3, 3};
    Py_BEGIN_ALLOW_THREADS
    for (int i = 0; i < 3; i++) {
        ht_vdec_stream stream;
        ht_vdec_picture picture;
        memset(&stream, 0, sizeof stream);
        memset(&picture, 0, sizeof picture);
        stream.end_of_frame = 1; stream.need_display = 1;
        stream.len = (uint32_t)spec[i].sz;
        picture.width = (uint32_t)spec[i].w; picture.height = (uint32_t)spec[i].h;
        picture.width_stride = (uint32_t)spec[i].ws;
        picture.height_stride = (uint32_t)spec[i].hs;
        picture.pixel_format = (uint32_t)spec[i].fmt;
        picture.buffer_size = (uint32_t)spec[i].out;

        status[i] = n_malloc(dev, &source[i], spec[i].sz);
        if (status[i] != 0) { phase[i] = 0; continue; }
        status[i] = n_memcpy(source[i], (size_t)spec[i].sz,
                             (const void *)(uintptr_t)spec[i].ja, (size_t)spec[i].sz, 1);
        if (status[i] != 0) {
            phase[i] = 1; n_free(source[i]); source[i] = NULL; continue;
        }
        stream.addr = (uint64_t)(uintptr_t)source[i];
        status[i] = n_malloc(dev, &output[i], spec[i].out);
        if (status[i] != 0) {
            phase[i] = 2; n_free(source[i]); source[i] = NULL; continue;
        }
        picture.vir_addr = (uint64_t)(uintptr_t)output[i];
        status[i] = n_send((uint32_t)spec[i].chn, &stream, &picture, (uint32_t)timeout);
        if (status[i] != 0) phase[i] = 3;  /* caller frees both, drops channel */
    }
    Py_END_ALLOW_THREADS
    PyObject *list = PyList_New(3);
    if (list == NULL) return NULL;
    for (int i = 0; i < 3; i++) {
        PyObject *item = Py_BuildValue("iiKK", status[i], phase[i],
                                       (unsigned long long)(uintptr_t)source[i],
                                       (unsigned long long)(uintptr_t)output[i]);
        if (item == NULL) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, item);
    }
    return list;
}

/* finish(chn, out_size, dst_addr, stream_id, timeout_ms) ->
 *   (status, phase, pixel_format, frame_flag,
 *    va0, va1, va2, ws0, ws1, ws2, hs0, hs1, hs2)
 * phases: 0 get_frame failed (no frame held)  1 memcpy_async failed
 *         2 release_frame failed  3 ok.  Whenever get_frame succeeded the
 * frame is released exactly once inside this call, as _finish does. */
static PyObject *af_finish(PyObject *self, PyObject *args) {
    (void)self;
    int chn, timeout;
    unsigned long long out_size, dst_addr, stream_id;
    if (!PyArg_ParseTuple(args, "iKKKi", &chn, &out_size, &dst_addr,
                          &stream_id, &timeout)) return NULL;
    if (n_get_frame == NULL || n_release == NULL || n_memcpy_async == NULL)
        Py_RETURN_NONE;

    ht_video_frame frame;
    uint64_t out2[8], out3[8];      /* >= the binding's 16/48-byte outs */
    memset(&frame, 0, sizeof frame);
    memset(out2, 0, sizeof out2); memset(out3, 0, sizeof out3);

    int status = 0, phase = 3;
    Py_BEGIN_ALLOW_THREADS
    status = n_get_frame((uint32_t)chn, &frame, out2, out3, (uint32_t)timeout);
    if (status != 0) { phase = 0; goto out; }
    status = n_memcpy_async((void *)(uintptr_t)dst_addr, (size_t)out_size,
                            (const void *)(uintptr_t)frame.virt_addr[0],
                            (size_t)out_size, 3 /*device->device*/,
                            (void *)(uintptr_t)stream_id);
    if (status != 0) { phase = 1; }
    {
        int rel = n_release((uint32_t)chn, &frame);
        if (status == 0 && rel != 0) { status = rel; phase = 2; }
    }
out: ;
    Py_END_ALLOW_THREADS
    return Py_BuildValue("iiIIKKKIIIIII", status, phase,
                         frame.pixel_format, frame.frame_flag,
                         (unsigned long long)frame.virt_addr[0],
                         (unsigned long long)frame.virt_addr[1],
                         (unsigned long long)frame.virt_addr[2],
                         frame.width_stride[0], frame.width_stride[1],
                         frame.width_stride[2],
                         frame.height_stride[0], frame.height_stride[1],
                         frame.height_stride[2]);
}

/* free_batch(sequence of device pointers) -> count freed */
static PyObject *af_free_batch(PyObject *self, PyObject *obj) {
    (void)self;
    if (n_free == NULL) Py_RETURN_NONE;
    PyObject *fast = PySequence_Fast(obj, "free_batch expects a sequence");
    if (fast == NULL) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n > 64) n = 64;                    /* a wave holds <= 3 pairs */
    void *ptrs[64];
    for (Py_ssize_t i = 0; i < n; i++) {
        unsigned long long p = PyLong_AsUnsignedLongLong(
            PySequence_Fast_GET_ITEM(fast, i));
        if (PyErr_Occurred()) { Py_DECREF(fast); return NULL; }
        ptrs[i] = (void *)(uintptr_t)p;
    }
    Py_DECREF(fast);
    Py_ssize_t count = 0;
    if (n > 0) {
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t i = 0; i < n; i++) if (n_free(ptrs[i]) == 0) count++;
        Py_END_ALLOW_THREADS
    }
    return PyLong_FromSsize_t(count);
}

/* malloc(dev, size) -> (status, ptr); memcpy / memcpy_async -> status
 * (parity helpers for probes and the non-fast fallback comparisons) */
static PyObject *af_malloc(PyObject *self, PyObject *args) {
    (void)self;
    int dev; unsigned long long size;
    if (!PyArg_ParseTuple(args, "iK", &dev, &size)) return NULL;
    if (n_malloc == NULL) Py_RETURN_NONE;
    void *p = NULL; int status;
    Py_BEGIN_ALLOW_THREADS
    status = n_malloc(dev, &p, size);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("iK", status, (unsigned long long)(uintptr_t)p);
}

static PyObject *af_memcpy(PyObject *self, PyObject *args) {
    (void)self;
    unsigned long long dst, src, count; int kind;
    if (!PyArg_ParseTuple(args, "KKKi", &dst, &src, &count, &kind)) return NULL;
    if (n_memcpy == NULL) Py_RETURN_NONE;
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = n_memcpy((void *)(uintptr_t)dst, (size_t)count,
                      (const void *)(uintptr_t)src, (size_t)count, kind);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(status);
}

static PyObject *af_memcpy_async(PyObject *self, PyObject *args) {
    (void)self;
    unsigned long long dst, src, count, stream; int kind;
    if (!PyArg_ParseTuple(args, "KKKiK", &dst, &src, &count, &kind, &stream))
        return NULL;
    if (n_memcpy_async == NULL) Py_RETURN_NONE;
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = n_memcpy_async((void *)(uintptr_t)dst, (size_t)count,
                            (const void *)(uintptr_t)src, (size_t)count, kind,
                            (void *)(uintptr_t)stream);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(status);
}

static PyMethodDef methods[] = {
    {"init", af_init, METH_NOARGS, "resolve natives; True when usable"},
    {"probe", af_probe, METH_VARARGS, "dvpp_get_image_info under released GIL"},
    {"start", af_start, METH_VARARGS,
     "malloc+memcpy+malloc+send_stream under released GIL"},
    {"start3", af_start3, METH_VARARGS,
     "three start chains under one released-GIL pass"},
    {"finish", af_finish, METH_VARARGS,
     "get_frame+memcpy_async+release_frame under released GIL"},
    {"free_batch", af_free_batch, METH_O, "dvpp_free a batch of device pointers"},
    {"malloc", af_malloc, METH_VARARGS, "dvpp_malloc parity helper"},
    {"memcpy", af_memcpy, METH_VARARGS, "aclrtMemcpy parity helper"},
    {"memcpy_async", af_memcpy_async, METH_VARARGS, "aclrtMemcpyAsync parity helper"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_aclfast",
    "GIL-releasing wrappers for the hot CANN DVPP host calls",
    -1, methods, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__aclfast(void) {
    return PyModule_Create(&module);
}

/* GIL-releasing wire decode helpers for xinfer (r11 knife A).
 *
 * base64.b64decode, bytes(buffer) and np.require's read-only copy are single C calls
 * that hold the GIL for their whole execution, so the wall_oss device path's view-pool
 * parallelism was nominal: its tickets reached the engine's device-image thread only
 * ~2 ms into a 3.3 ms prepare (probe-timeline-r10). These helpers run the same decodes
 * and copies under Py_BEGIN_ALLOW_THREADS; anything non-canonical is declined (None)
 * and the caller falls back to the stdlib call, so accepted results are equal by
 * construction. Built per platform (see the wall_oss r11 notes); xinfer falls back to
 * the stdlib paths when the module is absent.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

static unsigned char b64_value[256];

static void b64_init(void) {
    memset(b64_value, 0xFF, sizeof(b64_value));
    for (int i = 0; i < 26; i++) { b64_value['A' + i] = (unsigned char)i; }
    for (int i = 0; i < 26; i++) { b64_value['a' + i] = (unsigned char)(26 + i); }
    for (int i = 0; i < 10; i++) { b64_value['0' + i] = (unsigned char)(52 + i); }
    b64_value['+'] = 62;
    b64_value['/'] = 63;
}

/* b64decode(text) -> bytes | None: canonical standard base64 only.
 *
 * Declines (returns None) for any non-canonical shape: length not a multiple of 4,
 * alphabet characters outside the strict set, padding anywhere but the very end,
 * non-ASCII input. The caller then falls back to base64.b64decode, whose validate=False
 * semantics (discarding non-alphabet bytes) we deliberately do not reproduce.
 */
static PyObject *wirefast_b64decode(PyObject *self, PyObject *arg) {
    (void)self;
    if (PyUnicode_Check(arg)) {
        Py_ssize_t n = 0;
        const char *s = PyUnicode_AsUTF8AndSize(arg, &n);
        if (s == NULL) return NULL;
        /* b64_value initialized in PyInit__wirefast */
        Py_ssize_t pad = 0;
        int ok = (n % 4 == 0);
        if (ok && n >= 1 && s[n - 1] == '=') pad++;
        if (ok && n >= 2 && s[n - 2] == '=') pad++;
        for (Py_ssize_t i = 0; ok && i < n - pad; i++) {
            unsigned char c = (unsigned char)s[i];
            if (c >= 128 || b64_value[c] == 0xFF) ok = 0;
        }
        if (ok) {
            Py_ssize_t quads = n / 4;
            Py_ssize_t out_len = 3 * quads - pad;
            if (out_len < 0) out_len = 0;
            PyObject *out = PyBytes_FromStringAndSize(NULL, out_len);
            if (out == NULL) return NULL;
            if (out_len > 0) {
                unsigned char *dst = (unsigned char *)PyBytes_AS_STRING(out);
                const unsigned char *src = (const unsigned char *)s;
                Py_BEGIN_ALLOW_THREADS
                for (Py_ssize_t q = 0; q < quads; q++) {
                    const unsigned char *p = src + 4 * q;
                    unsigned trip = ((unsigned)b64_value[p[0]] << 18)
                                  | ((unsigned)b64_value[p[1]] << 12)
                                  | ((unsigned)b64_value[p[2]] << 6)
                                  |  (unsigned)b64_value[p[3]];
                    if (q + 1 == quads) {
                        dst[0] = (unsigned char)(trip >> 16);
                        if (pad != 2) dst[1] = (unsigned char)(trip >> 8);
                        if (pad == 0) dst[2] = (unsigned char)trip;
                    } else {
                        dst[0] = (unsigned char)(trip >> 16);
                        dst[1] = (unsigned char)(trip >> 8);
                        dst[2] = (unsigned char)trip;
                    }
                    dst += 3;
                }
                Py_END_ALLOW_THREADS
            }
            return out;
        }
    }
    Py_RETURN_NONE;
}

/* to_bytearray(buffer) -> bytearray | None: one GIL-released copy of any simple buffer.
 *
 * bytes, bytearray, memoryview and C-contiguous arrays all provide a PyBUF_SIMPLE
 * buffer; the copy runs with the GIL released so the requesting thread's prepare work
 * continues during multi-MB frame copies. A buffer that cannot be exported simply is
 * declined and the caller falls back.
 */
static PyObject *wirefast_to_bytearray(PyObject *self, PyObject *arg) {
    Py_buffer view;
    (void)self;
    if (PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) == 0) {
        Py_ssize_t n = view.len;
        PyObject *out = PyByteArray_FromStringAndSize(NULL, n < 0 ? 0 : n);
        if (out != NULL) {
            char *dst = PyByteArray_AS_STRING(out);
            Py_BEGIN_ALLOW_THREADS
            if (n > 0) memcpy(dst, view.buf, (size_t)n);
            Py_END_ALLOW_THREADS
        }
        PyBuffer_Release(&view);
        return out;  /* NULL only on allocation failure, which raises */
    }
    PyErr_Clear();
    Py_RETURN_NONE;
}

static PyMethodDef wirefast_methods[] = {
    {"b64decode", wirefast_b64decode, METH_O,
     "b64decode(text: str) -> bytes | None -- canonical base64 decoded with the GIL released."},
    {"to_bytearray", wirefast_to_bytearray, METH_O,
     "to_bytearray(buffer) -> bytearray | None -- one GIL-released copy of a simple buffer."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef wirefast_module = {
    PyModuleDef_HEAD_INIT,
    "_wirefast",
    "GIL-releasing wire decode helpers (canonical inputs only; None = use the stdlib fallback).",
    -1,
    wirefast_methods,
};

PyMODINIT_FUNC PyInit__wirefast(void) {
    b64_init();
    return PyModule_Create(&wirefast_module);
}

#define PY_SSIZE_T_CLEAN
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>

/* Task-local fault injection: a conforming ndarray data allocator whose shrink moves.
 * Retired bytes are poisoned and kept allocated until reset, so stale reads are observable
 * without dereferencing actually freed memory. No service process imports this module. */
static PyDataMem_Handler original, probe;
static PyObject *previous;
static void *allocated, *retired;
static size_t allocated_size, retired_size;
static int fail_resize, moves;

static void *probe_malloc(void *ctx, size_t size) {
    (void)ctx;
    allocated = original.allocator.malloc(original.allocator.ctx, size);
    allocated_size = size;
    return allocated;
}
static void *probe_calloc(void *ctx, size_t count, size_t size) {
    (void)ctx;
    return original.allocator.calloc(original.allocator.ctx, count, size);
}
static void probe_free(void *ctx, void *ptr, size_t size) {
    (void)ctx;
    original.allocator.free(original.allocator.ctx, ptr, size);
}
static void *probe_realloc(void *ctx, void *ptr, size_t size) {
    (void)ctx;
    if (fail_resize) return NULL;
    if (ptr != allocated || size > allocated_size)
        return original.allocator.realloc(original.allocator.ctx, ptr, size);
    void *fresh = original.allocator.malloc(original.allocator.ctx, size);
    if (fresh == NULL) return NULL;
    memcpy(fresh, ptr, size);
    memset(ptr, 0, allocated_size);
    retired = ptr;
    retired_size = allocated_size;
    allocated = fresh;
    allocated_size = size;
    moves++;
    return fresh;
}
static PyObject *install(PyObject *self, PyObject *arg) {
    (void)self;
    if (previous != NULL) return PyErr_Format(PyExc_RuntimeError, "already installed");
    fail_resize = PyObject_IsTrue(arg);
    moves = 0;
    previous = PyDataMem_GetHandler();
    if (previous == NULL) return NULL;
    original = *(PyDataMem_Handler *)PyCapsule_GetPointer(previous, "mem_handler");
    memset(&probe, 0, sizeof(probe));
    strcpy(probe.name, "xinfer-host-review-moving-resize");
    probe.version = 1;
    probe.allocator = (PyDataMemAllocator){NULL, probe_malloc, probe_calloc, probe_realloc, probe_free};
    PyObject *capsule = PyCapsule_New(&probe, "mem_handler", NULL);
    if (capsule == NULL) return NULL;
    PyObject *old = PyDataMem_SetHandler(capsule);
    Py_DECREF(capsule);
    if (old == NULL) return NULL;
    Py_DECREF(old);
    Py_RETURN_NONE;
}
static PyObject *reset(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    PyObject *old = PyDataMem_SetHandler(previous);
    if (old == NULL) return NULL;
    Py_DECREF(old);
    Py_CLEAR(previous);
    if (retired != NULL) {
        original.allocator.free(original.allocator.ctx, retired, retired_size);
        retired = NULL;
    }
    return PyLong_FromLong(moves);
}
static PyMethodDef methods[] = {
    {"install", install, METH_O, NULL}, {"reset", reset, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "host_allocator_probe", NULL, -1, methods};
PyMODINIT_FUNC PyInit_host_allocator_probe(void) { import_array(); return PyModule_Create(&module); }

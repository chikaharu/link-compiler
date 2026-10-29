/* Minimal model of a numpy-1.x-compiled C extension's binary-layout gate.
 * check():  compares runtime descr size against a compile-time constant,
 *           exactly like the Cython/numpy check embedded in pandas .so files
 *           ("numpy.dtype size changed, may indicate binary incompatibility.
 *             Expected %zd from C header, got %zd from PyObject").
 * consume(): reads `elsize` at the *1.x offset* (32, int32) -- code that
 *           trusts the old layout, the way real extensions do inline.
 * The expected size lives in a struct right after a magic marker so the
 * built .so can be binary-patched: that patch IS the layout forgery. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

typedef struct { char magic[8]; int expected; } GateCfg;
static GateCfg cfg __attribute__((used)) = { "FORGE96!", 96 };

static PyObject* check(PyObject* self, PyObject* arg) {
    /* arg is the descr *type* itself (np.dtype): its tp_basicsize is the
     * struct size the C header was compiled against -- the value pandas'
     * embedded check compares (96 under 1.x, 88 under 2.x). */
    Py_ssize_t got = ((PyTypeObject*)arg)->tp_basicsize;
    if (got != cfg.expected) {
        PyErr_Format(PyExc_ValueError,
            "numpy.dtype size changed, may indicate binary incompatibility. "
            "Expected %d from C header, got %zd from PyObject", cfg.expected, got);
        return NULL;
    }
    return PyLong_FromSsize_t(got);
}

static PyObject* consume(PyObject* self, PyObject* arg) {
    int elsize_1x;  /* 1.x layout: elsize is int32 at offset 32 */
    memcpy(&elsize_1x, (const char*)arg + 32, sizeof(int));
    return PyLong_FromLong(elsize_1x);
}

static PyObject* magic(PyObject* self, PyObject* ignored) {
    return PyUnicode_FromStringAndSize(cfg.magic, 8);
}
static PyMethodDef methods[] = {
    {"magic", magic, METH_NOARGS, "marker (keeps cfg in the binary)"},
    {"check", check, METH_O, "layout gate"},
    {"consume", consume, METH_O, "read elsize at 1.x offset"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef mod = { PyModuleDef_HEAD_INIT, "gatelib", NULL, -1, methods };
PyMODINIT_FUNC PyInit_gatelib(void) { return PyModule_Create(&mod); }

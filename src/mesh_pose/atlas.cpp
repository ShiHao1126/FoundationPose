#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cmath>
#include <cstdint>
#include <limits>

// Inputs are contiguous float32 RGBA textures and row-major homogeneous poses.
// Return a new bytearray with aligned, writable storage for a float32 NCHW array.
static PyObject* pack(PyObject*, PyObject* args) {
    Py_buffer rgba{}, xyz{}, poses{};
    int count, columns, height, width, normalize;
    float radius;
    if (!PyArg_ParseTuple(args, "y*y*y*iiiipf", &rgba, &xyz, &poses, &count, &columns,
                          &height, &width, &normalize, &radius)) return nullptr;
    PyObject* result = nullptr;
    // Check each texture dimension against the actual storage before computing offsets.
    const int64_t pixels = int64_t(height) * width;
    const int64_t rows = columns > 0 ? (int64_t(count) + columns - 1) / columns : 0;
    const int64_t tiles = rows * columns;
    if (height <= 0 || width <= 0 || count <= 0 || columns <= 0 || radius <= 0 ||
        tiles > rgba.len / 16 / pixels || xyz.len != rgba.len ||
        poses.len != int64_t(count) * 16 * sizeof(float) ||
        count > PY_SSIZE_T_MAX / 24 / pixels) {
        PyErr_SetString(PyExc_ValueError, "Invalid atlas dimensions or pose buffer");
    } else {
        result = PyByteArray_FromStringAndSize(nullptr, count * pixels * 24);
    }
    if (result) {
        auto color = static_cast<const float*>(rgba.buf);
        auto coords = static_cast<const float*>(xyz.buf);
        auto matrices = static_cast<const float*>(poses.buf);
        auto output = reinterpret_cast<float*>(PyByteArray_AS_STRING(result));
        int bright = 0;
        Py_BEGIN_ALLOW_THREADS
        #pragma omp parallel for num_threads(4) schedule(static) reduction(|:bright)
        for (int i = 0; i < count; ++i) {
            float* dst = output + i * 6 * pixels;
            const float* matrix = matrices + int64_t(i) * 16;
            const int64_t left = int64_t(i % columns) * width;
            const int64_t top = int64_t(i / columns) * height;
            for (int y = 0; y < height; ++y) {
                const int64_t row = ((top + height - 1 - y) * columns * width + left) * 4;
                for (int x = 0; x < width; ++x) {
                    const int64_t src = row + int64_t(x) * 4;
                    const int64_t dest = int64_t(y) * width + x;
                    const float alpha = fminf(1.f, fmaxf(0.f, color[src + 3]));
                    for (int c = 0; c < 3; ++c) {
                        float rgb = (color[src + c] * alpha) * 255.f;
                        dst[c * pixels + dest] = rgb;
                        bright |= int(rgb > 1.f);
                        float value = coords[src + c] - matrix[c * 4 + 3];
                        if (normalize) {
                            value /= radius;
                            if (coords[src + 2] < .001f || std::abs(value) >= 2.f) value = 0.f;
                        }
                        dst[(c + 3) * pixels + dest] = value;
                    }
                }
            }
        }
        if (bright) {
            #pragma omp parallel for num_threads(4) schedule(static)
            for (int i = 0; i < count; ++i) {
                float* dst = output + i * 6 * pixels;
                for (int64_t j = 0; j < 3 * pixels; ++j) dst[j] /= 255.f;
            }
        }
        Py_END_ALLOW_THREADS
    }
    PyBuffer_Release(&poses);
    PyBuffer_Release(&xyz);
    PyBuffer_Release(&rgba);
    return result;
}

static PyMethodDef methods[] = {{"pack", pack, METH_VARARGS, "Pack rendered RGB-XYZ tiles."},
                                {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_atlas", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__atlas() { return PyModule_Create(&module); }

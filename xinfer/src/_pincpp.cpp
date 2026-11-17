/* PREPARE-C PC3: GIL-releasing batch pin_memory for xinfer prepare.
 *
 * HOST-GAP K5 pinned the ~15 prepared CPU tensors inside prepare so admission
 * only uploads; on the K1-K6 arm that pin loop measured 0.51 ms per request of
 * main-thread GIL-held torch calls (probe start.gilphase seg_pin, 2026-11-15)
 * -- the single largest main-thread GIL item left in the prepare window, and it
 * serializes the view-pool workers and the engine device-image thread behind it.
 * pin_batch runs the same ATen pin_memory() per CPU non-pinned tensor with the
 * GIL released, in dict-in dict-out form; a per-tensor failure (deviceless
 * frontend process) keeps the original tensor, exactly like the python path's
 * guarded fallback. Values, dtypes, shapes and freshness are identical: the
 * same op on the same inputs, only the GIL schedule changes.
 */
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

static std::string g_last_error;  // diagnostic of the most recent per-tensor fallback

py::dict pin_batch(py::dict tensors) {
    py::dict out;
    g_last_error.clear();
    /* py::isinstance<torch::Tensor> is always false for an unregistered pybind
     * typeinfo; check against the real torch.Tensor type object instead. */
    py::type tensor_type = py::module_::import("torch").attr("Tensor");
    for (auto item : tensors) {
        py::object value = py::reinterpret_borrow<py::object>(item.second);
        if (py::isinstance(value, tensor_type)) {
            torch::Tensor tensor = py::cast<torch::Tensor>(value);
            if (tensor.defined() && tensor.is_cpu() && !tensor.is_pinned()) {
                torch::Tensor pinned;
                try {
                    py::gil_scoped_release nogil;
                    pinned = tensor.pin_memory();
                } catch (const c10::Error &error) {
                    g_last_error = error.what();
                }
                if (pinned.defined()) {
                    if (pinned.is_pinned()) {
                        out[item.first] = pinned;
                        continue;
                    }
                    g_last_error = "pin_memory returned a non-pinned tensor";
                }
            }
        }
        out[item.first] = value;
    }
    return out;
}

std::string last_error() { return g_last_error; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pin_batch", &pin_batch,
          "pin_batch(dict) -> dict: every CPU non-pinned tensor pinned with the GIL released; "
          "failures and non-tensors pass through unchanged");
    m.def("last_error", &last_error, "the most recent per-tensor pin failure, for diagnostics");
}

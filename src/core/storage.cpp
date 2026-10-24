#include "backends/backend.h"

#include <stdexcept>

namespace dl::detail {

const Backend& backend(Device device) {
    if (device.type == DeviceType::cpu) {
        return cpu_backend();
    }
#ifdef CTORCH_USE_CUDA
    return cuda_backend();
#else
    throw std::runtime_error("Ctorch was built without CUDA support");
#endif
}

Storage::Storage(std::size_t bytes, Device device)
    : allocation(backend(device).allocate(bytes, device)), bytes(bytes), device(device) {}

} // namespace dl::detail

namespace dl {

void synchronize(Device device) { detail::backend(device).synchronize(device); }

} // namespace dl

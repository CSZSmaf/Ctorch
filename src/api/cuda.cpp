#include "api/common.h"

extern "C" {

dl_status dl_cuda_memory_get_stats(std::int32_t device, dl_cuda_memory_stats* output) {
    return dl::capi::invoke([&] {
        const auto stats = dl::cuda::memory_stats(device);
        *output = {stats.used_bytes, stats.reserved_bytes, stats.peak_used_bytes,
            stats.peak_reserved_bytes, stats.release_threshold};
    });
}

dl_status dl_cuda_memory_set_release_threshold(std::int32_t device, std::uint64_t bytes) {
    return dl::capi::invoke([&] { dl::cuda::set_memory_release_threshold(bytes, device); });
}

dl_status dl_cuda_memory_reset_peak_stats(std::int32_t device) {
    return dl::capi::invoke([&] { dl::cuda::reset_peak_memory_stats(device); });
}

dl_status dl_cuda_memory_release_unused(std::int32_t device) {
    return dl::capi::invoke([&] { dl::cuda::release_unused_memory(device); });
}

} // extern "C"

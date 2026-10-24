#include "dl/amp.h"

#include <array>

namespace dl {
namespace {
struct PrecisionState { DType dtype = DType::bf16; bool enabled = false; };
thread_local std::array<PrecisionState, 2> precision;
PrecisionState& state(DeviceType device) { return precision[static_cast<std::size_t>(device)]; }
}

bool is_autocast_enabled(DeviceType device) { return state(device).enabled; }
DType autocast_dtype(DeviceType device) { return state(device).dtype; }

AutocastGuard::AutocastGuard(DeviceType device, DType dtype, bool enabled)
    : device_(device), previous_dtype_(state(device).dtype), previous_enabled_(state(device).enabled) {
    state(device) = {dtype, enabled};
}
AutocastGuard::~AutocastGuard() { state(device_) = {previous_dtype_, previous_enabled_}; }

} // namespace dl

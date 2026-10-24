#include "dl/data.h"

#include "core/tensor_impl.h"
#include "dl/transfer.h"
#include "ops/numeric.h"

#include <deque>

namespace dl::data {
namespace {

Tensor snapshot(const Tensor& tensor) {
    auto value = tensor.detach();
    return value.device() == Device::cpu() ? value.clone() : value.to(Device::cpu());
}

struct Slot {
    std::optional<Batch> host;
    std::optional<Transfer> input, target;
    bool input_staged = false, target_staged = false;

    void stage(Batch batch, Device device) {
        if (input) { input->wait(); target->wait(); }
        input.reset();
        target.reset();
        auto prepare = [&](const Tensor& tensor, const Tensor* previous, bool& staged) {
            // 连续页锁定输入直接借用；只有预取器自己建立的暂存副本进入后续复用。
            if (device.type == DeviceType::cpu || tensor.device().type != DeviceType::cpu
                || (tensor.is_pinned() && tensor.is_contiguous())) {
                staged = false;
                return tensor;
            }
            if (staged && !tensor.requires_grad()
                && previous->shape() == tensor.shape() && previous->dtype() == tensor.dtype()) {
                auto buffer = *previous;
                detail::numeric::copy_into(detail::TensorAccess::mutable_data(buffer), detail::data_of(tensor));
                return buffer;
            }
            staged = !tensor.requires_grad();
            return tensor.pin_memory();
        };
        auto x = prepare(batch.input, host ? &host->input : nullptr, input_staged);
        auto y = prepare(batch.target, host ? &host->target : nullptr, target_staged);
        host = Batch{std::move(x), std::move(y)};
        input.emplace(host->input, device);
        target.emplace(host->target, device);
    }

    Batch result() const { return {input->tensor(), target->tensor()}; }

    Batch saved() const {
        // 自有暂存值在 pending 阶段保持不变；借用源则以已经传输的目标值为准。
        return {snapshot(input_staged ? host->input : input->tensor()),
                snapshot(target_staged ? host->target : target->tensor())};
    }
};

} // namespace

struct PrefetchLoader::Impl {
    Impl(DataLoader& source, Device device, std::size_t depth) : loader(source), device(device), depth(depth) {}

    void enqueue(Batch batch) {
        auto slot = free.empty() ? std::make_unique<Slot>() : std::move(free.front());
        if (!free.empty()) free.pop_front();
        slot->stage(std::move(batch), device);
        pending.push_back(std::move(slot));
    }

    DataLoader& loader;
    Device device;
    std::size_t depth;
    std::deque<std::unique_ptr<Slot>> pending, free;
};

PrefetchLoader::PrefetchLoader(DataLoader& source, Device device, std::size_t depth)
    : impl_(std::make_unique<Impl>(source, device, depth)) {}
PrefetchLoader::~PrefetchLoader() = default;
DataLoader& PrefetchLoader::source() const { return impl_->loader; }

std::optional<Batch> PrefetchLoader::next() {
    while (impl_->pending.size() < impl_->depth) {
        auto batch = impl_->loader.next();
        if (!batch) break;
        impl_->enqueue(std::move(*batch));
    }
    if (impl_->pending.empty()) return std::nullopt;
    auto result = impl_->pending.front()->result();
    impl_->free.push_back(std::move(impl_->pending.front()));
    impl_->pending.pop_front();
    return result;
}

void PrefetchLoader::reset() {
    impl_->pending.clear();
    impl_->free.clear();
    impl_->loader.reset();
}

PrefetchState PrefetchLoader::state_dict() const {
    PrefetchState result{impl_->depth, {}};
    result.pending.reserve(impl_->pending.size());
    for (const auto& slot : impl_->pending) result.pending.push_back(slot->saved());
    return result;
}

void PrefetchLoader::load_state_dict(const PrefetchState& state) {
    impl_->pending.clear();
    impl_->free.clear();
    impl_->depth = state.depth;
    for (const auto& batch : state.pending) impl_->enqueue(batch);
}

} // namespace dl::data

#include "dl/data.h"

#include "core/tensor_impl.h"
#include "dl/transfer.h"
#include "ops/numeric.h"

#include <condition_variable>
#include <deque>
#include <thread>

namespace dl::data {
namespace {

Tensor snapshot(const Tensor& tensor) {
    auto value = tensor.detach();
    return value.device() == Device::cpu() ? value.clone() : value.to(Device::cpu());
}

struct Field {
    std::optional<Tensor> host;
    std::optional<Transfer> transfer;
    bool staged = false;

    void stage(const Tensor& tensor, Device device) {
        if (transfer) transfer->wait();
        transfer.reset();
        // 连续页锁定输入直接借用；自建暂存副本在传输完成后复用。
        if (device.type == DeviceType::cpu || tensor.device().type != DeviceType::cpu
            || (tensor.is_pinned() && tensor.is_contiguous())) {
            host = tensor;
            staged = false;
        } else if (staged && !tensor.requires_grad() && host->shape() == tensor.shape() && host->dtype() == tensor.dtype()) {
            detail::numeric::copy_into(detail::TensorAccess::mutable_data(*host), detail::data_of(tensor));
        } else {
            host = tensor.pin_memory();
            staged = !tensor.requires_grad();
        }
        transfer.emplace(*host, device);
    }

    Tensor result(bool save) const {
        if (!save) return transfer->tensor();
        return snapshot(staged ? *host : transfer->tensor());
    }
};

struct Slot {
    std::vector<Field> fields;

    void stage(const Batch& batch, Device device) {
        fields.resize(2 + batch.extra.size());
        fields[0].stage(batch.input, device);
        fields[1].stage(batch.target, device);
        for (std::size_t i = 0; i < batch.extra.size(); ++i) fields[i + 2].stage(batch.extra[i], device);
    }

    Batch result(bool save = false) const {
        Batch batch{fields[0].result(save), fields[1].result(save), {}};
        for (std::size_t i = 2; i < fields.size(); ++i) batch.extra.push_back(fields[i].result(save));
        return batch;
    }
};

} // namespace

struct PrefetchLoader::Impl {
    Impl(DataLoader& source, Device device, std::size_t depth, bool background)
        : loader(source), device(device), depth(depth), background(background) {}
    ~Impl() { pause(); }

    void start() {
        if (!background || worker.joinable() || done) return;
        worker = std::jthread([this](std::stop_token stop) {
            try {
                std::unique_lock lock(mutex);
                while (changed.wait(lock, stop, [&] { return !ready; })) {
                    lock.unlock();
                    auto batch = loader.next();
                    lock.lock();
                    if (!batch) { done = true; changed.notify_all(); return; }
                    ready = std::move(batch);
                    changed.notify_all();
                }
            } catch (...) {
                std::lock_guard lock(mutex);
                failure = std::current_exception();
                changed.notify_all();
            }
        });
    }

    // 完成正在整理的批次，再停止线程。ready 保留该批次及其对应的采样位置。
    void pause() {
        if (!worker.joinable()) return;
        worker.request_stop();
        changed.notify_all();
        worker.join();
    }

    std::optional<Batch> take() {
        if (!background) return loader.next();
        start();
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return ready || done || failure; });
        if (failure) std::rethrow_exception(failure);
        auto batch = std::move(ready);
        ready.reset();
        changed.notify_all();
        return batch;
    }

    void enqueue(const Batch& batch) {
        auto slot = free.empty() ? std::make_unique<Slot>() : std::move(free.front());
        if (!free.empty()) free.pop_front();
        slot->stage(batch, device);
        pending.push_back(std::move(slot));
    }

    void clear() {
        pause();
        pending.clear();
        free.clear();
        ready.reset();
        done = false;
        failure = nullptr;
    }

    DataLoader& loader;
    Device device;
    std::size_t depth;
    bool background, done = false;
    std::deque<std::unique_ptr<Slot>> pending, free;
    std::optional<Batch> ready;
    std::exception_ptr failure;
    std::mutex mutex;
    std::condition_variable_any changed;
    std::jthread worker;
};

PrefetchLoader::PrefetchLoader(DataLoader& source, Device device, std::size_t depth, bool background)
    : impl_(std::make_unique<Impl>(source, device, depth, background)) {}
PrefetchLoader::~PrefetchLoader() = default;
DataLoader& PrefetchLoader::source() const { return impl_->loader; }

std::optional<Batch> PrefetchLoader::next() {
    impl_->start();
    while (impl_->pending.size() < impl_->depth) {
        auto batch = impl_->take();
        if (!batch) break;
        impl_->enqueue(*batch);
    }
    if (impl_->pending.empty()) return std::nullopt;
    auto result = impl_->pending.front()->result();
    impl_->free.push_back(std::move(impl_->pending.front()));
    impl_->pending.pop_front();
    return result;
}

void PrefetchLoader::reset() {
    impl_->clear();
    impl_->loader.reset();
}

void PrefetchLoader::reset(std::vector<Index> order) {
    impl_->clear();
    impl_->loader.reset(std::move(order));
}

PrefetchState PrefetchLoader::state_dict() const {
    impl_->pause();
    if (impl_->failure) std::rethrow_exception(impl_->failure);
    PrefetchState result{impl_->depth, {}, impl_->background};
    for (const auto& slot : impl_->pending) result.pending.push_back(slot->result(true));
    if (impl_->ready) {
        const auto& batch = *impl_->ready;
        Batch saved{snapshot(batch.input), snapshot(batch.target), {}};
        for (const auto& tensor : batch.extra) saved.extra.push_back(snapshot(tensor));
        result.pending.push_back(std::move(saved));
    }
    return result;
}

void PrefetchLoader::load_state_dict(const PrefetchState& state) {
    impl_->clear();
    impl_->depth = state.depth;
    impl_->background = state.background;
    for (const auto& batch : state.pending) impl_->enqueue(batch);
}

} // namespace dl::data

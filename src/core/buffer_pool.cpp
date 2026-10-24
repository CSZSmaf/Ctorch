#include "core/buffer_pool.h"
#include "dl/memory.h"

#include <algorithm>
#include <atomic>
#include <utility>

namespace dl::detail {
namespace {

thread_local BufferPool* current_pool = nullptr;

} // namespace

struct BufferPool::Block {
    Block(std::size_t bytes, Device device) : storage(bytes, device) {}
    Storage storage;
    std::atomic<bool> in_use{false};
};

struct BufferPool::Lease {
    explicit Lease(std::shared_ptr<Block> block) : block(std::move(block)) {}
    // 最后一个引用可在消费线程释放；归还发布先前访问，下一次借用以 acquire 接收。
    ~Lease() { block->in_use.store(false, std::memory_order_release); }
    std::shared_ptr<Block> block;
};

void BufferPool::reserve(std::size_t bytes, Device device, std::size_t count) {
    auto& buffers = buffers_[{device.type, device.index, bytes}];
    buffers.reserve(count);
    while (buffers.size() < count) buffers.push_back(std::make_shared<Block>(bytes, device));
}

std::shared_ptr<Storage> BufferPool::acquire(std::size_t bytes, Device device) {
    auto& buffers = buffers_[{device.type, device.index, bytes}];
    const auto unused = std::find_if(buffers.begin(), buffers.end(),
        [](const auto& block) { return !block->in_use.load(std::memory_order_acquire); });
    auto block = unused == buffers.end()
        ? buffers.emplace_back(std::make_shared<Block>(bytes, device)) : *unused;
    auto lease = std::make_shared<Lease>(block);
    block->in_use.store(true, std::memory_order_relaxed);
    // 别名 shared_ptr 保留普通 Storage 访问，控制块负责归还与底层块的所有权。
    return {std::move(lease), &block->storage};
}

BufferPool::Stats BufferPool::stats(Device device) const {
    Stats result;
    for (const auto& [key, buffers] : buffers_) {
        if (std::get<0>(key) != device.type || std::get<1>(key) != device.index) continue;
        result.buffers += buffers.size();
        for (const auto& block : buffers) {
            result.reserved_bytes += block->storage.bytes;
            if (block->in_use.load(std::memory_order_acquire)) {
                ++result.in_use_buffers;
                result.in_use_bytes += block->storage.bytes;
            }
        }
    }
    return result;
}

void BufferPool::release_unused() {
    for (auto it = buffers_.begin(); it != buffers_.end();) {
        std::erase_if(it->second, [](const auto& block) {
            return !block->in_use.load(std::memory_order_acquire);
        });
        if (it->second.empty()) it = buffers_.erase(it);
        else ++it;
    }
}

BufferPool* exchange_buffer_pool(BufferPool* pool) noexcept { return std::exchange(current_pool, pool); }

std::shared_ptr<Storage> allocate_storage(std::size_t bytes, Device device) {
    return current_pool ? current_pool->acquire(bytes, device) : std::make_shared<Storage>(bytes, device);
}

} // namespace dl::detail

namespace dl {

struct MemoryGuard::Impl {
    detail::BufferPool pool;
    detail::BufferPool* previous = detail::exchange_buffer_pool(&pool);
};

MemoryGuard::MemoryGuard() : impl_(std::make_unique<Impl>()) {}
MemoryGuard::~MemoryGuard() { detail::exchange_buffer_pool(impl_->previous); }

MemoryGuard::Stats MemoryGuard::memory_stats(Device device) const {
    const auto stats = impl_->pool.stats(device);
    return {stats.buffers, stats.in_use_buffers, stats.reserved_bytes, stats.in_use_bytes};
}

void MemoryGuard::release_unused_memory() { impl_->pool.release_unused(); }

} // namespace dl

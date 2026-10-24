#pragma once

#include "core/layout.h"

#include <memory>
#include <utility>

namespace dl::detail {

/// 分配结果携带释放函数与执行上下文，设备内存的归还沿同一 stream 排序。
struct Allocation {
    void* address;
    void (*release)(void*, void*) noexcept;
    std::shared_ptr<void> context;
};

/// 存储拥有原始内存；各个视图通过共享所有权延长其生命周期。
struct Storage {
    Storage(std::size_t bytes, Device device);
    Storage(Allocation allocation, std::size_t bytes, Device device, bool pinned = false, bool read_only = false)
        : allocation(std::move(allocation)), bytes(bytes), device(device), pinned(pinned), read_only(read_only) {}
    ~Storage() { allocation.release(allocation.address, allocation.context.get()); }
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    Allocation allocation;
    std::size_t bytes;
    Device device;
    bool pinned = false;
    // 外部只读存储供普通算子读取，视图保持该属性；clone 创建可写的独立存储。
    bool read_only = false;
};

/// 普通分配沿当前工作区选择可复用存储；无工作区时直接交给既有后端。
std::shared_ptr<Storage> allocate_storage(std::size_t bytes, Device device);

/// 数据描述独立于求导历史，反向节点保存它时只持有布局与存储。
struct TensorData {
    std::shared_ptr<Storage> storage;
    Layout layout;
    DType dtype;

    Device device() const { return storage->device; }

    const void* address() const {
        if (layout.numel() == 0) {
            return storage->allocation.address;
        }
        return static_cast<const std::byte*>(storage->allocation.address) + layout.offset * element_size(dtype);
    }

    void* mutable_address() {
        if (layout.numel() == 0) {
            return storage->allocation.address;
        }
        return static_cast<std::byte*>(storage->allocation.address) + layout.offset * element_size(dtype);
    }

    template <class T> const T* data() const { return static_cast<const T*>(address()); }
    template <class T> T* mutable_data() { return static_cast<T*>(mutable_address()); }
};

inline TensorData empty_data(Shape shape, DType dtype, Device device = Device::cpu()) {
    auto storage = allocate_storage(static_cast<std::size_t>(numel(shape)) * element_size(dtype), device);
    return {std::move(storage), contiguous_layout(std::move(shape)), dtype};
}

} // namespace dl::detail

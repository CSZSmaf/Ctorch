#pragma once

#include "core/tensor_data.h"

#include <map>
#include <tuple>

namespace dl::detail {

/// 工作区按设备和精确字节数持有原生 Storage，每次借用共享一个归还凭据。
/// 最后一个张量或保存值结束持有时归还块，原生后端上下文继续由 Storage 拥有。
class BufferPool {
public:
    BufferPool() = default;
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    struct Stats {
        std::size_t buffers = 0, in_use_buffers = 0;
        std::size_t reserved_bytes = 0, in_use_bytes = 0;
    };

    void reserve(std::size_t bytes, Device device, std::size_t count);
    std::shared_ptr<Storage> acquire(std::size_t bytes, Device device);
    Stats stats(Device device) const;
    void release_unused();

private:
    struct Block;
    struct Lease;
    using Key = std::tuple<DeviceType, int, std::size_t>;
    std::map<Key, std::vector<std::shared_ptr<Block>>> buffers_;
};

/// 切换当前线程的普通张量分配池，返回外层池供工作区的栈上作用域恢复。
/// nullptr 表示原有直接分配路径；同一池的获取、查询和清理由工作区串行安排。
BufferPool* exchange_buffer_pool(BufferPool* pool) noexcept;

} // namespace dl::detail

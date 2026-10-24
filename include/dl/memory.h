#pragma once

#include "dl/types.h"

#include <memory>

namespace dl {

/// 当前线程内的张量分配共用缓冲池，最后一个张量或反向保存值释放后复用存储。
/// 作用域可嵌套，退出时恢复外层池；输出仍可独立持有存储并在其他线程释放。
/// 对象在创建线程中按栈顺序销毁，每个并发执行线程使用自己的作用域。
class DL_API MemoryGuard {
public:
    MemoryGuard();
    ~MemoryGuard();
    MemoryGuard(const MemoryGuard&) = delete;
    MemoryGuard& operator=(const MemoryGuard&) = delete;

    struct Stats {
        std::size_t buffers = 0, in_use_buffers = 0;
        std::size_t reserved_bytes = 0, in_use_bytes = 0;
    };
    /// reserved 包含空闲缓冲，in_use 只计入仍由张量或保存值持有的缓冲。
    Stats memory_stats(Device device = Device::cpu()) const;
    /// 归还各设备的空闲缓冲，继续保留仍被使用的存储。
    void release_unused_memory();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dl

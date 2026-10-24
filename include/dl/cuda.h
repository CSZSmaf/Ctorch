#pragma once

#include "dl/types.h"

namespace dl::cuda {

/// 当前进程中该设备的 Ctorch 内存池统计，单位为字节。
/// used 使用 Runtime 的活跃分配口径，包含保留的工作区；reserved 包含池内空闲空间。
/// 外部导入存储、页锁定宿主内存和 CUDA/cuBLAS/cuDNN 自身分配不计入这些数值。
struct MemoryStats {
    std::uint64_t used_bytes, reserved_bytes;
    std::uint64_t peak_used_bytes, peak_reserved_bytes;
    std::uint64_t release_threshold;
};

/// 各属性直接从 Runtime 读取，不等待 stream；阶段边界可先 synchronize 再读取。
/// 首次调用建立指定设备的上下文，CPU-only 构建通过异常交付功能不可用。
DL_API MemoryStats memory_stats(int device = 0);
/// 同步点尝试释放超过阈值的池内空闲内存；默认零，阈值不是显存容量上限。
DL_API void set_memory_release_threshold(std::uint64_t bytes, int device = 0);
/// 重置 Runtime 的两个峰值属性，当前分配与保留空间继续由池持有。
DL_API void reset_peak_memory_stats(int device = 0);
/// 归还缓存工作区，等待计算与复制 stream，再尝试释放池内全部空闲空间。
/// 在训练阶段之间调用；存活张量、外部存储和当前保留阈值继续保持原样。
DL_API void release_unused_memory(int device = 0);

} // namespace dl::cuda

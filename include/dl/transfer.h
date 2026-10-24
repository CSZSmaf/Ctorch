#pragma once

#include "dl/tensor.h"

namespace dl {

/// 一次传输拥有源缓冲、目标和完成事件。对象销毁时等待尚未完成的复制。
/// CPU/CUDA 之间使用独立复制 stream；同设备直接共享，跨 GPU 沿用同步传输。
/// 页锁定 CPU 输入和连续 CUDA 输入由传输共享，wait 完成前保持其内容不变。
class DL_API Transfer {
public:
    Transfer(const Tensor& input, Device target);
    ~Transfer();
    Transfer(Transfer&&) noexcept;
    Transfer& operator=(Transfer&&) noexcept;
    Transfer(const Transfer&) = delete;
    Transfer& operator=(const Transfer&) = delete;

    /// CUDA 结果通过事件接入计算 stream；CPU 结果在复制完成后交付。
    Tensor tensor() const;
    bool ready() const;
    void wait() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dl

#pragma once

#include "dl/types.h"

#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace dl {
namespace detail {
struct TensorImpl;
struct TensorAccess;
} // namespace detail

/// 共享张量身份的轻量句柄。复制句柄保留身份，视图另建布局并共享存储。
class DL_API Tensor {
public:
    const Shape& shape() const;
    const Shape& strides() const;
    Index size(Index axis) const;
    Index ndim() const;
    Index numel() const;
    /// 相对所属存储起点的元素偏移；data() 返回的地址已经包含此偏移。
    Index storage_offset() const;
    DType dtype() const;
    Device device() const;
    bool is_contiguous() const;
    /// 外部存储可声明为只读；参数更新、初始化和状态加载使用可写存储。
    /// clone 创建独立可写副本，普通计算和视图直接使用原有存储。
    bool is_read_only() const;
    bool requires_grad() const;
    bool is_leaf() const;

    /// 在完整前向图之间切换浮点叶张量的求导状态，所有同身份句柄一起生效。
    /// 关闭时释放叶梯度槽；重新开启时建立空槽，数据、布局和身份保持原样。
    /// 相同状态的重复设置保留现有梯度；非叶结果先 detach 得到新的叶身份。
    void requires_grad_(bool enabled = true) const;

    /// 返回当前设备上第一个逻辑元素的只读地址；T 对应张量 dtype。
    /// 连续张量可顺序读取 numel() 个元素，视图的后续位置由 stride 描述。
    /// CPU 地址用于宿主读取，CUDA 地址用于设备接口；item 与 to_vector 提供宿主副本。
    template <class T> const T* data() const { return static_cast<const T*>(raw_data()); }

    template <class T> T item() const { return *detach().to(Device::cpu()).data<T>(); }

    /// 按逻辑行主序复制到宿主容器，保持原张量的数据与求导历史。
    template <class T> std::vector<T> to_vector() const {
        if (numel() == 0) {
            return {};
        }
        auto dense = detach().to(Device::cpu()).contiguous();
        auto first = dense.data<T>();
        return std::vector<T>(first, first + numel());
    }

    Tensor clone() const;
    Tensor detach() const;
    Tensor contiguous() const;
    Tensor astype(DType dtype) const;
    /// 同设备返回当前句柄；跨设备复制连续数据，并将梯度传回原设备。
    Tensor to(Device device) const;
    /// 将 CPU 张量复制到页锁定内存；已有连续页锁定张量直接返回自身。
    /// 该内存用于异步 CUDA 传输，创建它需要启用 CUDA 的构建。
    Tensor pin_memory() const;
    bool is_pinned() const;

    /// view 使用现有布局可表达的形状；reshape 在需要时复制逻辑数据。
    /// shape 中可用一个 -1 表示由元素总数与其余非零维度确定的维度。
    Tensor view(Shape shape) const;
    Tensor reshape(Shape shape) const;
    Tensor transpose(Index first, Index second) const;
    Tensor permute(Shape axes) const;

    /// 在 [-rank-1, rank] 范围插入单例轴，保留数据存储与偏移。
    Tensor unsqueeze(Index axis) const;
    /// 移除全部单例轴，或仅移除指定单例轴；标量保持标量。
    Tensor squeeze() const;
    Tensor squeeze(Index axis) const;
    Tensor flatten(Index first = 0, Index last = -1) const;

    /// 选择轴上的半开区间 [start, stop)，step 为正，端点位于该轴范围内。
    Tensor slice(Index axis, Index start, Index stop, Index step = 1) const;
    Tensor expand(Shape shape) const;
    /// select 移除所选轴，narrow 保留长度为 length 的区间；位置可从轴尾负向计数。
    Tensor select(Index axis, Index index) const;
    Tensor narrow(Index axis, Index start, Index length) const;
    /// 沿轴返回视图列表；split_size、chunks 为正，显式长度非负且总和等于轴长。
    std::vector<Tensor> split(Index split_size, Index axis = 0) const;
    std::vector<Tensor> split(Shape sizes, Index axis = 0) const;
    std::vector<Tensor> chunk(Index chunks, Index axis = 0) const;
    std::vector<Tensor> unbind(Index axis = 0) const;
    /// repeats 非负且长度不少于 rank；新增前导轴按长度一理解，结果拥有独立数据。
    Tensor repeat(Shape repeats) const;

    /// 标量反向使用种子 1；显式 seed 与输出的 shape 和 dtype 相同。
    /// 一次反向消费该前向图，叶梯度在不同前向图之间累计。
    void backward() const;
    void backward(const Tensor& seed) const;
    std::optional<Tensor> grad() const;
    void zero_grad() const;

private:
    explicit Tensor(std::shared_ptr<detail::TensorImpl> impl) : impl_(std::move(impl)) {}
    const void* raw_data() const;

    std::shared_ptr<detail::TensorImpl> impl_;
    friend struct detail::TensorAccess;
};

DL_API Tensor full(Shape shape, double value, TensorOptions options = {});
DL_API Tensor zeros(Shape shape, TensorOptions options = {});
DL_API Tensor ones(Shape shape, TensorOptions options = {});

/// like 工厂继承形状、dtype 与设备，创建独立的连续数据；求导属性由参数单独指定。
DL_API Tensor full_like(const Tensor& input, double value, bool requires_grad = false);
DL_API Tensor zeros_like(const Tensor& input, bool requires_grad = false);
DL_API Tensor ones_like(const Tensor& input, bool requires_grad = false);
/// 有限的 double 起点与步长描述半开区间；步长非零且朝向终点，元素数可由 Index 表示。
/// 与其他工厂一样默认 float32，整数索引通过 options.dtype=i64 指定。
DL_API Tensor arange(double end, TensorOptions options = {});
DL_API Tensor arange(double start, double end, double step = 1.0, TensorOptions options = {});
/// steps 非负；steps=1 时返回 start，其余非空序列包含两端点。
DL_API Tensor linspace(double start, double end, Index steps, TensorOptions options = {});
DL_API Tensor eye(Index rows, TensorOptions options = {});
DL_API Tensor eye(Index rows, Index columns, TensorOptions options = {});

namespace detail {
DL_API Tensor from_host(const void* data, Shape shape, DType dtype, bool requires_grad, Device device);
}

/// 将连续宿主数据复制为独立张量，values 的长度等于 shape 的元素数。
template <class T>
Tensor from_span(std::span<const T> values, Shape shape, bool requires_grad = false,
                 Device device = Device::cpu()) {
    return detail::from_host(values.data(), std::move(shape), dtype_of<T>, requires_grad, device);
}

} // namespace dl

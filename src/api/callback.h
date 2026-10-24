#pragma once

#include "api/common.h"

namespace dl::capi {

/// 描述、执行闭包和求导节点共用一份回调状态，最后一个拥有者执行可选释放函数。
inline std::shared_ptr<void> callback_state(void* value, dl_user_data_deleter deleter) {
    return {value, [deleter](void* value) { if (deleter) deleter(value); }};
}

/// 为一次同步 C 回调建立稳定的借用句柄，张量身份和存储继续由普通 Tensor 持有。
class BorrowedTensors {
public:
    explicit BorrowedTensors(std::span<const Tensor> values) {
        handles_.reserve(values.size());
        pointers_.reserve(values.size());
        for (const auto& value : values) {
            handles_.push_back({value});
            pointers_.push_back(&handles_.back());
        }
    }
    BorrowedTensors(const BorrowedTensors&) = delete;
    BorrowedTensors& operator=(const BorrowedTensors&) = delete;
    const dl_tensor* const* data() const { return pointers_.data(); }
    std::size_t size() const { return pointers_.size(); }

private:
    std::vector<dl_tensor> handles_;
    std::vector<const dl_tensor*> pointers_;
};

/// 单个拥有型结果在回调期间交付，提取值之后由栈对象结束句柄的持有。
template <class Handle>
struct ReturnedHandle {
    ReturnedHandle() = default;
    ~ReturnedHandle() { delete value; }
    ReturnedHandle(const ReturnedHandle&) = delete;
    ReturnedHandle& operator=(const ReturnedHandle&) = delete;
    Handle* value = nullptr;
};

/// 梯度槽按输入数量建立，空槽表示没有贡献；已交付句柄随本次桥接调用释放。
struct ReturnedTensors {
    explicit ReturnedTensors(std::size_t count) : values(count, nullptr) {}
    ~ReturnedTensors() { for (auto* tensor : values) delete tensor; }
    ReturnedTensors(const ReturnedTensors&) = delete;
    ReturnedTensors& operator=(const ReturnedTensors&) = delete;
    std::vector<dl_tensor*> values;
};

} // namespace dl::capi

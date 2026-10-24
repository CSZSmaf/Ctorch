#pragma once

#include "autograd/graph.h"
#include "dl/tensor.h"

namespace dl::detail {

struct TensorImpl {
    TensorImpl(TensorData data, std::unique_ptr<AutogradMeta> autograd)
        : data(std::move(data)), autograd(std::move(autograd)) {}

    TensorData data;
    std::unique_ptr<AutogradMeta> autograd;
};

/// 框架内部的写入入口集中在创建器、状态加载和优化器中。
struct TensorAccess {
    static const TensorData& data(const Tensor& tensor) { return tensor.impl_->data; }
    static TensorData& mutable_data(Tensor& tensor) { return tensor.impl_->data; }
    static const AutogradMeta* autograd(const Tensor& tensor) { return tensor.impl_->autograd.get(); }
    static const void* identity(const Tensor& tensor) { return tensor.impl_.get(); }

    static Tensor wrap(TensorData data, std::unique_ptr<AutogradMeta> autograd = {}) {
        return Tensor(std::make_shared<TensorImpl>(std::move(data), std::move(autograd)));
    }
};

inline Tensor make_tensor(TensorData data, bool requires_grad = false) {
    return TensorAccess::wrap(std::move(data), requires_grad ? make_leaf_meta() : nullptr);
}

inline const TensorData& data_of(const Tensor& tensor) { return TensorAccess::data(tensor); }

} // namespace dl::detail

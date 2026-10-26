#pragma once

#include "dl/c_api.h"
#include "dl/dl.h"

#include <exception>

struct dl_tensor { dl::Tensor value; };
struct dl_graph { dl::plan::Graph value; };
struct dl_tensor_list { std::vector<dl::Tensor> value; };
struct dl_gradients { dl::Gradients value; };
struct dl_module { std::shared_ptr<dl::nn::UnaryModule> value; };
struct dl_functional {
    std::shared_ptr<dl::nn::UnaryModule> owner;
    dl::nn::FunctionalModule value;
    explicit dl_functional(std::shared_ptr<dl::nn::UnaryModule> module)
        : owner(std::move(module)), value(*owner) {}
};
struct dl_functional_state { dl::nn::FunctionalState value; };
struct dl_optimizer { std::shared_ptr<dl::optim::Optimizer> value; };
struct dl_scheduler {
    std::shared_ptr<dl::optim::Optimizer> optimizer;
    std::unique_ptr<dl::optim::LRScheduler> value;
};
struct dl_generator { std::shared_ptr<dl::Generator> value; };
struct dl_state { dl::StateDict value; };
struct dl_checkpoint { dl::io::TrainingCheckpoint value; };
struct dl_grad_scope {
    explicit dl_grad_scope(bool enabled) : value(enabled) {}
    dl::GradModeGuard value;
};
struct dl_autocast_scope { dl::AutocastGuard value; };
struct dl_grad_scaler { dl::GradScaler value; };
struct dl_loader {
    // 成员按逆序析构，采样器先于它借用的显式随机生成器结束生命周期。
    std::shared_ptr<dl::Generator> generator;
    std::shared_ptr<dl::data::DataLoader> value;
};
struct dl_transfer { dl::Transfer value; };
struct dl_prefetch_loader {
    std::shared_ptr<dl::Generator> generator;
    std::shared_ptr<dl::data::DataLoader> source;
    std::unique_ptr<dl::data::PrefetchLoader> value;
};

namespace dl::capi {

// 独立扩展库共用主库的线程错误槽，C 调用方统一通过 dl_last_error 读取。
DL_API void set_error(const char* message) noexcept;
const char* error_text() noexcept;

/// 语言边界统一交付异常结果，错误文本写入固定线程缓冲，不参与计算路径的分配。
template <class Function> dl_status invoke(Function&& function) noexcept {
    try {
        function();
        set_error("");
        return DL_OK;
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("unknown C++ exception");
    }
    return DL_ERROR;
}

template <class Handle, class Function> dl_status make_handle(Handle** output, Function&& function) noexcept {
    *output = nullptr;
    return invoke([&] { *output = new Handle{function()}; });
}

inline Device convert(dl_device value) { return {static_cast<DeviceType>(value.type), value.index}; }
inline RandomKey convert(dl_random_key value) { return {value.seed, value.stream}; }
inline dl_random_key convert(RandomKey value) { return {value.seed, value.stream}; }
inline dl_device convert(Device value) { return {static_cast<dl_device_type>(value.type), value.index}; }
inline DType convert(dl_dtype value) { return static_cast<DType>(value); }
inline LossReduction convert(dl_loss_reduction value) { return static_cast<LossReduction>(value); }
inline InterpolationMode convert(dl_interpolation_mode value) { return static_cast<InterpolationMode>(value); }
inline RNNActivation convert(dl_rnn_activation value) { return static_cast<RNNActivation>(value); }
inline Shape convert(dl_shape value) {
    return value.size == 0 ? Shape{} : Shape(value.data, value.data + value.size);
}
inline TensorSpec convert(dl_tensor_spec value) {
    return {convert(value.shape), convert(value.strides), convert(value.dtype), convert(value.device),
        value.storage_offset, value.read_only != 0};
}
inline dl_tensor_spec convert(const TensorSpec& value) {
    return {{value.shape.data(), value.shape.size()}, {value.strides.data(), value.strides.size()},
        static_cast<dl_dtype>(value.dtype), convert(value.device), value.storage_offset, value.read_only};
}
inline TensorOptions convert(dl_tensor_options value) {
    return {convert(value.dtype), convert(value.device), value.requires_grad != 0};
}
inline Size2 convert(dl_size2 value) { return {value.height, value.width}; }
inline Conv1dOptions convert(dl_conv1d_options value) {
    return {value.stride, value.padding, value.dilation, value.groups};
}
inline Pool1dOptions convert(dl_pool1d_options value) { return {value.kernel, value.stride, value.padding}; }
inline Conv2dOptions convert(dl_conv2d_options value) {
    return {convert(value.stride), convert(value.padding), convert(value.dilation), value.groups};
}
inline Pool2dOptions convert(dl_pool2d_options value) {
    return {convert(value.kernel), convert(value.stride), convert(value.padding)};
}
inline BatchNormOptions convert(dl_batch_norm_options value) {
    return {value.epsilon, value.momentum, value.affine != 0, value.track_running_stats != 0};
}
inline AttentionOptions convert(dl_attention_options value) { return {value.dropout, value.bias != 0}; }
inline AttentionComputeOptions convert(dl_attention_compute_options value) {
    return {value.dropout, value.causal != 0, static_cast<AttentionBackend>(value.backend)};
}
inline RecurrentOptions convert(dl_recurrent_options value) {
    return {value.layers, value.bias != 0, value.batch_first != 0, value.bidirectional != 0, value.dropout};
}
inline optim::SGDOptions convert(dl_sgd_options value) {
    return {value.learning_rate, value.momentum, value.dampening, value.weight_decay, value.nesterov != 0};
}
inline optim::AdamWOptions convert(dl_adamw_options value) {
    return {value.learning_rate, value.beta1, value.beta2, value.epsilon, value.weight_decay};
}
inline optim::AdamOptions convert(dl_adam_options value) {
    return {value.learning_rate, value.beta1, value.beta2, value.epsilon, value.weight_decay};
}
inline optim::RMSpropOptions convert(dl_rmsprop_options value) {
    return {value.learning_rate, value.alpha, value.epsilon, value.weight_decay, value.momentum, value.centered != 0};
}
inline optim::AdagradOptions convert(dl_adagrad_options value) {
    return {value.learning_rate, value.learning_rate_decay, value.weight_decay, value.initial_accumulator, value.epsilon};
}
inline data::LoaderOptions convert(dl_loader_options value) {
    return {value.batch_size, value.shuffle != 0, value.drop_last != 0};
}
inline std::optional<Tensor> optional(const dl_tensor* tensor) {
    return tensor ? std::optional{tensor->value} : std::nullopt;
}
inline Generator& random_generator(dl_generator* generator) {
    return generator ? *generator->value : default_generator();
}

std::vector<Tensor> tensor_list(const dl_tensor* const* tensors, std::size_t count);
DL_API std::filesystem::path utf8_path(const char* text);

template <class Function>
dl_status unary(const dl_tensor* input, dl_tensor** output, Function function) {
    return make_handle(output, [&] { return function(input->value); });
}

template <class Function>
dl_status binary(const dl_tensor* a, const dl_tensor* b, dl_tensor** output, Function function) {
    return make_handle(output, [&] { return function(a->value, b->value); });
}

} // namespace dl::capi

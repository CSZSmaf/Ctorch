#include "dl/plan.h"

#include "core/layout.h"
#include "dl/ops.h"
#include "dl/random.h"

#include <algorithm>

namespace dl::plan::ops {
namespace {

OutputSpec dense(const TensorSpec& input) {
    return {contiguous_spec(input.shape, accumulation_dtype(input.dtype), input.device), std::nullopt};
}

OutputSpec dropout_spec(const TensorSpec& input, double probability, bool training) {
    if (!training || probability == 0.0) return {input, 0};
    // p == 1 的 where 分支沿用存储 dtype，其余训练分支以累积 dtype 计算。
    if (probability == 1.0) return {contiguous_spec(input.shape, input.dtype, input.device)};
    return dense(input);
}

TensorSpec key_spec() { return contiguous_spec({2}, DType::i64); }

Operation binary(std::string name, Tensor (*execute)(const Tensor&, const Tensor&)) {
    return {std::move(name), [](std::span<const TensorSpec> inputs) {
        return OutputSpec{contiguous_spec(detail::broadcast_shape(inputs[0].shape, inputs[1].shape),
            accumulation_dtype(inputs[0].dtype), inputs[0].device), std::nullopt};
    }, [execute](std::span<const Tensor> inputs) { return execute(inputs[0], inputs[1]); }};
}

Operation unary(std::string name, Tensor (*execute)(const Tensor&)) {
    return {std::move(name), [](std::span<const TensorSpec> inputs) { return dense(inputs[0]); },
        [execute](std::span<const Tensor> inputs) { return execute(inputs[0]); }};
}

struct BatchInputs {
    std::optional<BatchNormState> state;
    std::optional<Tensor> weight, bias;
};

BatchInputs batch_inputs(std::span<const Tensor> inputs, const BatchNormOptions& options) {
    BatchInputs result;
    if (options.track_running_stats) result.state = BatchNormState{inputs[1], inputs[2]};
    if (options.affine) {
        const std::size_t offset = options.track_running_stats ? 3 : 1;
        result.weight = inputs[offset];
        result.bias = inputs[offset + 1];
    }
    return result;
}

} // namespace

Operation add() {
    auto operation = binary("add", dl::add);
    operation.semantic = Semantic::add;
    return operation;
}
Operation add_relu() { return binary("add_relu", dl::add_relu); }
Operation mul() { return binary("mul", dl::mul); }
Operation relu() {
    auto operation = unary("relu", dl::relu);
    operation.semantic = Semantic::relu;
    return operation;
}
Operation silu() { return unary("silu", dl::silu); }

Operation attention(AttentionComputeOptions options, bool masked, bool keyed) {
    return {"attention", [options](std::span<const TensorSpec> inputs) {
        auto shape = inputs[0].shape;
        if (options.backend == AttentionBackend::math) {
            auto prefix = [](const TensorSpec& input) {
                return std::span<const Index>(input.shape).first(input.shape.size() - 2);
            };
            shape = detail::broadcast_shape(prefix(inputs[0]), prefix(inputs[1]));
            shape = detail::broadcast_shape(shape, prefix(inputs[2]));
            shape.push_back(inputs[0].shape[inputs[0].shape.size() - 2]);
            shape.push_back(inputs[2].shape.back());
        } else {
            shape.back() = inputs[2].shape.back();
        }
        const auto dtype = options.backend == AttentionBackend::cudnn
            ? DType::f32 : accumulation_dtype(inputs[0].dtype);
        return OutputSpec{contiguous_spec(std::move(shape), dtype, inputs[0].device)};
    }, [options, masked, keyed](std::span<const Tensor> inputs) {
        const auto mask = masked ? std::optional{inputs[3]} : std::nullopt;
        return keyed
            ? dl::scaled_dot_product_attention(options, inputs[0], inputs[1], inputs[2],
                key_from_tensor(inputs[3 + static_cast<std::size_t>(masked)]), mask)
            : dl::scaled_dot_product_attention(options, inputs[0], inputs[1], inputs[2], mask);
    }, !keyed && options.dropout > 0.0 && options.dropout < 1.0 ? Effect::random : Effect::none};
}

Operation matmul() {
    return {"matmul", [](std::span<const TensorSpec> inputs) {
        const auto& a = inputs[0];
        const auto& b = inputs[1];
        const auto left = std::span<const Index>(a.shape), right = std::span<const Index>(b.shape);
        const bool left_vector = left.size() == 1, right_vector = right.size() == 1;
        auto shape = detail::broadcast_shape(left.first(left_vector ? 0 : left.size() - 2),
            right.first(right_vector ? 0 : right.size() - 2));
        if (!left_vector) shape.push_back(left[left.size() - 2]);
        if (!right_vector) shape.push_back(right.back());
        return OutputSpec{contiguous_spec(std::move(shape), accumulation_dtype(a.dtype), a.device), std::nullopt};
    }, [](std::span<const Tensor> inputs) { return dl::matmul(inputs[0], inputs[1]); }};
}

Operation linear() {
    return {"linear", [](std::span<const TensorSpec> inputs) {
        auto shape = inputs[0].shape;
        shape.back() = inputs[1].shape[0];
        return OutputSpec{contiguous_spec(std::move(shape), accumulation_dtype(inputs[0].dtype), inputs[0].device)};
    }, [](std::span<const Tensor> inputs) {
        const auto bias = inputs.size() == 3 ? std::optional{inputs[2]} : std::nullopt;
        return dl::linear(inputs[0], inputs[1], bias);
    }};
}

Operation reshape(Shape shape) {
    return {"reshape", [shape](std::span<const TensorSpec> inputs) {
        auto result = inputs[0];
        auto target = detail::resolve_shape(shape, result.numel());
        auto strides = detail::reshape_strides({result.shape, result.strides, result.storage_offset}, target);
        if (!strides) return OutputSpec{contiguous_spec(std::move(target), result.dtype, result.device), std::nullopt};
        result.shape = std::move(target);
        result.strides = std::move(*strides);
        return OutputSpec{std::move(result), 0};
    }, [shape](std::span<const Tensor> inputs) { return inputs[0].reshape(shape); }};
}

Operation transpose(Index first, Index second) {
    return {"transpose", [first, second](std::span<const TensorSpec> inputs) {
        auto result = inputs[0];
        const auto rank = static_cast<Index>(result.shape.size());
        const auto a = detail::axis_index(first, rank), b = detail::axis_index(second, rank);
        std::swap(result.shape[a], result.shape[b]);
        std::swap(result.strides[a], result.strides[b]);
        return OutputSpec{std::move(result), 0};
    }, [first, second](std::span<const Tensor> inputs) { return inputs[0].transpose(first, second); }};
}

Operation contiguous() {
    return {"contiguous", [](std::span<const TensorSpec> inputs) {
        const auto& input = inputs[0];
        return input.is_contiguous() ? OutputSpec{input, 0}
            : OutputSpec{contiguous_spec(input.shape, input.dtype, input.device), std::nullopt};
    }, [](std::span<const Tensor> inputs) { return inputs[0].contiguous(); }};
}

Operation dropout(double probability, bool training) {
    const bool identity = !training || probability == 0.0;
    return {"dropout", [probability, training](std::span<const TensorSpec> inputs) {
        return dropout_spec(inputs[0], probability, training);
    }, [probability, training](std::span<const Tensor> inputs) {
        return dl::dropout(inputs[0], probability, training);
    }, !identity && probability < 1.0 ? Effect::random : Effect::none};
}

Operation batch_norm(BatchNormOptions options, bool training) {
    return {"batch_norm", [](std::span<const TensorSpec> inputs) { return dense(inputs[0]); },
        [options, training](std::span<const Tensor> inputs) {
            const auto bound = batch_inputs(inputs, options);
            if (bound.state) {
                return dl::batch_norm_(inputs[0], *bound.state, bound.weight, bound.bias,
                    training, options.momentum, options.epsilon);
            }
            return dl::batch_norm(inputs[0], std::nullopt, bound.weight, bound.bias,
                training, options.momentum, options.epsilon).output;
        }, training && options.track_running_stats ? Effect::mutation : Effect::none};
}

MultiOperation functional_batch_norm(BatchNormOptions options, bool training) {
    return {"functional_batch_norm", [options, training](std::span<const TensorSpec> inputs) {
        std::vector<OutputSpec> outputs{dense(inputs[0])};
        if (options.track_running_stats) {
            for (std::size_t i = 1; i <= 2; ++i) {
                outputs.push_back(training
                    ? OutputSpec{contiguous_spec(inputs[i].shape, inputs[i].dtype, inputs[i].device)}
                    : OutputSpec{inputs[i], i});
            }
        }
        return outputs;
    }, [options, training](std::span<const Tensor> inputs) {
        const auto bound = batch_inputs(inputs, options);
        auto result = dl::batch_norm(inputs[0], bound.state, bound.weight, bound.bias,
            training, options.momentum, options.epsilon);
        std::vector<Tensor> outputs{std::move(result.output)};
        if (result.state) {
            outputs.push_back(std::move(result.state->mean));
            outputs.push_back(std::move(result.state->variance));
        }
        return outputs;
    }};
}

MultiOperation unbind(Index axis) {
    return {"unbind", [axis](std::span<const TensorSpec> inputs) {
        auto element = inputs[0];
        const auto dim = detail::axis_index(axis, static_cast<Index>(element.shape.size()));
        const auto count = element.shape[dim], stride = element.strides[dim];
        element.shape.erase(element.shape.begin() + dim);
        element.strides.erase(element.strides.begin() + dim);
        std::vector<OutputSpec> outputs;
        outputs.reserve(static_cast<std::size_t>(count));
        for (Index index = 0; index < count; ++index) {
            auto spec = element;
            spec.storage_offset += index * stride;
            outputs.push_back({std::move(spec), 0});
        }
        return outputs;
    }, [axis](std::span<const Tensor> inputs) { return inputs[0].unbind(axis); }};
}

MultiOperation lstm_cell(bool bias) {
    return {"lstm_cell", [](std::span<const TensorSpec> inputs) {
        auto state = contiguous_spec(inputs[1].shape, accumulation_dtype(inputs[0].dtype), inputs[0].device);
        return std::vector<OutputSpec>{{state}, {std::move(state)}};
    }, [bias](std::span<const Tensor> inputs) {
        const auto bias_ih = bias ? std::optional{inputs[5]} : std::nullopt;
        const auto bias_hh = bias ? std::optional{inputs[6]} : std::nullopt;
        auto result = dl::lstm_cell(inputs[0], {inputs[1], inputs[2]}, inputs[3], inputs[4], bias_ih, bias_hh);
        return std::vector<Tensor>{std::move(result.hidden), std::move(result.cell)};
    }};
}

Operation fold_in_key(std::uint64_t label) {
    return {"fold_in_key", [](std::span<const TensorSpec>) { return OutputSpec{key_spec()}; },
        [label](std::span<const Tensor> inputs) { return key_tensor(dl::fold_in(key_from_tensor(inputs[0]), label)); }};
}

MultiOperation split_key(std::size_t count) {
    return {"split_key", [count](std::span<const TensorSpec>) {
        return std::vector<OutputSpec>(count, OutputSpec{key_spec()});
    }, [count](std::span<const Tensor> inputs) {
        const auto key = key_from_tensor(inputs[0]);
        std::vector<Tensor> outputs;
        outputs.reserve(count);
        for (std::size_t i = 0; i < count; ++i) outputs.push_back(key_tensor(dl::fold_in(key, i)));
        return outputs;
    }};
}

Operation uniform(Shape shape, double low, double high, TensorOptions options) {
    return {"uniform_with_key", [shape, options](std::span<const TensorSpec>) {
        return OutputSpec{contiguous_spec(shape, options.dtype, options.device)};
    }, [shape, low, high, options](std::span<const Tensor> inputs) {
        return dl::uniform(key_from_tensor(inputs[0]), shape, low, high, options);
    }};
}

Operation randn(Shape shape, TensorOptions options) {
    return {"randn_with_key", [shape, options](std::span<const TensorSpec>) {
        return OutputSpec{contiguous_spec(shape, options.dtype, options.device)};
    }, [shape, options](std::span<const Tensor> inputs) { return dl::randn(key_from_tensor(inputs[0]), shape, options); }};
}

Operation bernoulli(Shape shape, double probability, Device device) {
    return {"bernoulli_with_key", [shape, device](std::span<const TensorSpec>) {
        return OutputSpec{contiguous_spec(shape, DType::boolean, device)};
    }, [shape, probability, device](std::span<const Tensor> inputs) {
        return dl::bernoulli(key_from_tensor(inputs[0]), shape, probability, device);
    }};
}

Operation dropout_with_key(double probability, bool training) {
    return {"dropout_with_key", [probability, training](std::span<const TensorSpec> inputs) {
        return dropout_spec(inputs[0], probability, training);
    }, [probability, training](std::span<const Tensor> inputs) {
        return dl::dropout(inputs[0], probability, key_from_tensor(inputs[1]), training);
    }};
}

} // namespace dl::plan::ops

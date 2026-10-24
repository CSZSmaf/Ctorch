#include "dl/plan.h"

#include "optim/result_tensors.h"

namespace dl::plan::ops {
namespace {

std::vector<OutputSpec> counted_outputs(const TensorSpec& parameter, std::size_t values) {
    const auto spec = contiguous_spec(parameter.shape, parameter.dtype, parameter.device);
    std::vector<OutputSpec> outputs(values, OutputSpec{spec});
    outputs.push_back({contiguous_spec({}, DType::i64)});
    return outputs;
}

template <class Options, class Update>
MultiOperation adam_operation(const char* name, Options options, Update update) {
    return {name, [](std::span<const TensorSpec> inputs) {
        return counted_outputs(inputs[0], 3);
    }, [options, update](std::span<const Tensor> inputs) {
        std::optional<optim::AdamMoments> moments;
        if (inputs.size() == 5)
            moments = optim::AdamMoments{inputs[2], inputs[3], *inputs[4].data<Index>()};
        return optim::detail::result_tensors(update(inputs[0], inputs[1], moments, options));
    }};
}

} // namespace

MultiOperation sgd(optim::SGDOptions options) {
    return {"sgd", [options](std::span<const TensorSpec> inputs) {
        const auto& parameter = inputs[0];
        const auto spec = contiguous_spec(parameter.shape, parameter.dtype, parameter.device);
        std::vector<OutputSpec> outputs{{spec}};
        if (options.momentum != 0.0 || inputs.size() == 3) outputs.push_back({spec});
        return outputs;
    }, [options](std::span<const Tensor> inputs) {
        const auto velocity = inputs.size() == 3 ? std::optional{inputs[2]} : std::nullopt;
        return optim::detail::result_tensors(optim::sgd(inputs[0], inputs[1], velocity, options));
    }};
}

MultiOperation adamw(optim::AdamWOptions options) {
    return adam_operation("adamw", options, optim::adamw);
}

MultiOperation adam(optim::AdamOptions options) {
    return adam_operation("adam", options, optim::adam);
}

MultiOperation rmsprop(optim::RMSpropOptions options) {
    return {"rmsprop", [options](std::span<const TensorSpec> inputs) {
        return counted_outputs(inputs[0], 2 + (options.momentum != 0.0) + options.centered);
    }, [options](std::span<const Tensor> inputs) {
        std::optional<optim::RMSpropBuffers> buffers;
        if (inputs.size() != 2) {
            std::size_t position = 3;
            auto momentum = options.momentum != 0.0 ? std::optional{inputs[position++]} : std::nullopt;
            auto average = options.centered ? std::optional{inputs[position++]} : std::nullopt;
            buffers = optim::RMSpropBuffers{inputs[2], std::move(momentum), std::move(average),
                *inputs[position].data<Index>()};
        }
        return optim::detail::result_tensors(optim::rmsprop(inputs[0], inputs[1], buffers, options));
    }};
}

MultiOperation adagrad(optim::AdagradOptions options) {
    return {"adagrad", [](std::span<const TensorSpec> inputs) {
        return counted_outputs(inputs[0], 2);
    }, [options](std::span<const Tensor> inputs) {
        std::optional<optim::AdagradAccumulator> accumulator;
        if (inputs.size() == 4)
            accumulator = optim::AdagradAccumulator{inputs[2], *inputs[3].data<Index>()};
        return optim::detail::result_tensors(optim::adagrad(inputs[0], inputs[1], accumulator, options));
    }};
}

} // namespace dl::plan::ops

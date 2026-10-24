#include "dl/plan.h"

#include "dl/functional.h"

namespace dl::plan::ops {

MultiOperation functional_module(nn::FunctionalModule module, std::vector<std::uint8_t> training,
                                 MultiInfer infer, MultiExecute forward) {
    const auto parameter_count = module.parameter_names().size();
    const auto buffer_count = module.buffer_names().size();
    return {"functional_module",
        [buffer_count, infer = std::move(infer)](std::span<const TensorSpec> inputs) {
            const auto buffer_begin = inputs.size() - buffer_count - 1;
            std::vector<TensorSpec> bound_specs(inputs.begin(), inputs.end());
            std::vector<OutputSpec> outputs;
            outputs.reserve(buffer_count);
            for (std::size_t i = 0; i < buffer_count; ++i) {
                const auto& source = inputs[buffer_begin + i];
                bound_specs[buffer_begin + i] = contiguous_spec(source.shape, source.dtype, source.device);
                outputs.push_back({bound_specs[buffer_begin + i]});
            }
            auto results = infer(bound_specs);
            outputs.reserve(buffer_count + results.size());
            for (auto& result : results) {
                // 新 buffer 先成为图结果，前向返回其视图时沿同一存储组记录生命周期。
                if (result.alias_input && *result.alias_input >= buffer_begin
                    && *result.alias_input < buffer_begin + buffer_count) {
                    result.alias_output = *result.alias_input - buffer_begin;
                    result.alias_input.reset();
                } else if (result.alias_output) {
                    *result.alias_output += buffer_count;
                }
                outputs.push_back(std::move(result));
            }
            return outputs;
        },
        [module = std::move(module), training = std::move(training), forward = std::move(forward),
         parameter_count, buffer_count](std::span<const Tensor> inputs) {
            const auto buffer_begin = inputs.size() - buffer_count - 1;
            const auto parameter_begin = buffer_begin - parameter_count;
            auto result = module.call(inputs.subspan(parameter_begin, parameter_count),
                inputs.subspan(buffer_begin, buffer_count), training, key_from_tensor(inputs.back()),
                [&] { return forward(inputs.first(parameter_begin)); });
            result.buffers.reserve(buffer_count + result.output.size());
            for (auto& value : result.output) result.buffers.push_back(std::move(value));
            return std::move(result.buffers);
        }};
}

} // namespace dl::plan::ops

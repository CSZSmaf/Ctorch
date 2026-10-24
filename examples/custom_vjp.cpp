#include <dl/dl.h>

#include <array>
#include <iostream>

/// 新算子将规格推导、前向和 VJP 放在同一个普通 C++ 结构中。
struct ScaledSquare {
    double scale;

    dl::TensorSpec infer(const dl::TensorSpec& input) const {
        return dl::contiguous_spec(input.shape, dl::accumulation_dtype(input.dtype), input.device);
    }

    dl::Tensor operator()(const dl::Tensor& input) const {
        return dl::custom_vjp({input},
            [factor = scale](dl::VjpContext& context, std::span<const dl::Tensor> inputs) {
                context.save_for_backward({inputs[0]});
                return inputs[0] * inputs[0] * factor;
            },
            [factor = scale](const dl::VjpContext& context, const dl::Tensor& grad) {
                dl::VjpGradients contributions(context.input_count());
                if (context.needs_input_grad(0))
                    contributions[0] = grad * context.saved_tensors()[0] * (2.0 * factor);
                return contributions;
            });
    }
};

int main() {
    const std::array<float, 5> values{-2, -1, 0, 1, 2};
    auto input = dl::from_span<float>(values, {5}, true);
    ScaledSquare operation{0.5};
    auto specification = operation.infer(dl::tensor_spec(input));
    auto output = operation(input);
    dl::sum(output).backward();

    std::cout << "elements " << specification.numel() << "\noutput";
    for (auto value : output.to_vector<float>()) std::cout << ' ' << value;
    std::cout << "\ngradient";
    for (auto value : input.grad()->to_vector<float>()) std::cout << ' ' << value;
    std::cout << '\n';
}

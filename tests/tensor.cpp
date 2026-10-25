#include "test_support.h"

int main() {
    return test::run([] {
        auto x = test::tensor({1, 2, 3, 4, 5, 6}, {2, 3}, true);
        auto view = x.transpose(0, 1);
        test::expect(view.data<double>() == x.data<double>(), "transpose shares storage");
        test::expect(view.strides() == dl::Shape({1, 3}), "transpose strides");
        test::values(view.reshape({6}), {1, 4, 2, 5, 3, 6});
        auto bias = test::tensor({.5, 1, 1.5}, {3}, true);
        auto shared = x + bias;
        dl::sum(shared * shared + shared).backward();
        test::gradient(x, {4, 7, 10, 10, 13, 16});
        test::gradient(bias, {14, 20, 26});
        x.zero_grad();
        dl::sum(x * x).backward();
        test::gradient(x, {2, 4, 6, 8, 10, 12});

        for (auto dtype : {dl::DType::f16, dl::DType::bf16}) {
            auto input = test::tensor({1, 2, 3, 4, 5, 6}, {2, 3}).astype(dtype).detach();
            input.requires_grad_();
            auto factor = test::tensor({2, 4}, {2}).astype(dtype);
            auto product = input.transpose(0, 1) * factor;
            test::expect(product.dtype() == dl::DType::f32, "low precision arithmetic accumulates in float32");
            test::values(product, {2, 16, 4, 20, 6, 24});
            dl::sum(product).backward();
            test::gradient(input, {2, 2, 2, 4, 4, 4});
        }

        auto data = test::tensor({-2, 1, -1, 3}, {2, 2}, true);
        auto offset = test::tensor({1, -2}, {2}, true);
        dl::plan::Graph graph;
        auto a = graph.input(dl::tensor_spec(data));
        auto b = graph.input(dl::tensor_spec(offset));
        auto sum = graph.call(dl::plan::ops::add(), {a, b});
        auto output = graph.call(dl::plan::ops::relu(), {sum});
        auto plan = graph.compile({output});
        auto separate = graph.compile({output}, {.fuse_pointwise = false});
        test::expect(plan.fusion_count() == 1 && plan.step_count() == 1, "add/relu fusion");
        auto eager = dl::relu(data + offset);
        test::expect(plan({data, offset})[0].to_vector<double>() == eager.to_vector<double>(), "plan/eager output");
        test::expect(separate({data, offset})[0].to_vector<double>() == eager.to_vector<double>(), "unfused output");
        auto workspace = plan.prepare();
        dl::sum(workspace({data, offset})[0]).backward();
        test::gradient(data, {0, 0, 0, 1});
        test::gradient(offset, {0, 1});
        test::expect(dl::describe(view).find("strides=[1, 3]") != std::string::npos, "tensor metadata");
        test::expect(dl::describe(plan).find("fusions=1") != std::string::npos, "plan metadata");
    });
}

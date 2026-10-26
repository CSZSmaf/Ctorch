#include "test_support.h"

int main() {
    return test::run([] {
        auto matrix = dl::arange(6).reshape({2, 3});
        bool bad_shape = false, bad_view = false;
        try { matrix.reshape({-1, -1}); } catch (const std::invalid_argument&) { bad_shape = true; }
        try { matrix.transpose(0, 1).view({6}); } catch (const std::invalid_argument&) { bad_view = true; }
        test::expect(bad_shape && bad_view, "reshape and view report incompatible shapes");
        test::values(matrix.transpose(0, 1).reshape({6}), {0, 3, 1, 4, 2, 5});
        auto sequence = test::tensor({3, 1, 1, 2, 4, -1}, {2, 3}, true);
        auto sorted = dl::sort(sequence, 1);
        test::values(sorted.values, {1, 1, 3, -1, 2, 4});
        test::expect(sorted.indices.to_vector<dl::Index>() == dl::Shape({1, 2, 0, 2, 0, 1}), "stable sorting");
        dl::sum(dl::topk(sequence, 2, 1).values).backward();
        test::gradient(sequence, {1, 1, 0, 1, 1, 0});
        sequence.zero_grad();
        test::values(dl::cumsum(sequence.transpose(0, 1), 0), {3, 2, 4, 6, 5, 5});
        dl::sum(dl::cumsum(sequence, 1)).backward();
        test::gradient(sequence, {3, 2, 1, 3, 2, 1});
        test::values(dl::one_hot(test::indices({2, 0}), 3), {0, 0, 1, 1, 0, 0});

        auto scalar = dl::full({}, 1.5, {.dtype = dl::DType::f64, .requires_grad = true});
        test::close(scalar.item<float>(), 1.5);
        test::close(scalar.astype(dl::DType::f32).item<double>(), 1.5);
        test::expect(scalar.item<dl::Index>() == 1, "scalar numeric conversion");
        test::expect(scalar.requires_grad() && !scalar.grad(), "host reads preserve autograd state");
        auto converted = test::tensor({1.5, -2, 3, 4.5}, {2, 2}).transpose(0, 1);
        test::expect(converted.to_vector<float>() == std::vector<float>({1.5f, 3, -2, 4.5f}),
                     "host conversion follows logical layout");
        test::expect(dl::zeros({0}).to_vector<double>().empty(), "empty host conversion");
        auto large_integer = test::indices({9007199254740993LL});
        test::expect(large_integer.item<dl::Index>() == 9007199254740993LL, "integer extraction stays exact");
        for (auto dtype : {dl::DType::f16, dl::DType::bf16, dl::DType::boolean}) {
            auto one = dl::ones({}, {.dtype = dtype});
            test::close(one.item<float>(), 1);
            test::expect(one.to_vector<double>() == std::vector<double>({1}), "storage type conversion");
        }

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

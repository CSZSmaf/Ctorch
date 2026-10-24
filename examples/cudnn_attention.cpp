#include <dl/dl.h>

#include <iostream>
#include <optional>

int main() {
    dl::manual_seed(41);
    const dl::TensorOptions options{.device = dl::Device::cuda()};
    auto input = dl::randn({2, 64, 64}, options);
    auto target = input * 0.5;
    dl::nn::TransformerEncoder model(2, 64, 2, 128, 0.1, options);
    // 各层在融合图中计算因果关系，前向不再需要显式的方形 mask。
    model.set_attention_backend(dl::AttentionBackend::cudnn, true);
    dl::optim::AdamW optimizer(model.parameters(), {.learning_rate = 0.001});
    std::optional<dl::Tensor> last_loss;

    for (int step = 0; step < 8; ++step) {
        optimizer.zero_grad();
        auto prediction = [&] {
            // 主参数保持 FP32；投影与注意力按 BF16 输入计算，后续归一化继续使用 FP32。
            dl::AutocastGuard precision(dl::DeviceType::cuda, dl::DType::bf16);
            return model(input);
        }();
        last_loss = dl::mse_loss(prediction, target);
        last_loss->backward();
        optimizer.step();
    }
    // 只在这个示例训练阶段结束后读取一次设备标量。
    std::cout << "last training loss " << last_loss->item<float>() << '\n';
}

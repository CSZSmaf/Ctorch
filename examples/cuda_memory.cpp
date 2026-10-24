#include <dl/dl.h>

#include <iostream>
#include <string>

void print_memory(const char* phase, int device) {
    const auto stats = dl::cuda::memory_stats(device);
    std::cout << phase << ": used=" << stats.used_bytes << " reserved=" << stats.reserved_bytes
              << " peak_used=" << stats.peak_used_bytes << " peak_reserved=" << stats.peak_reserved_bytes
              << " threshold=" << stats.release_threshold << '\n';
}

int main(int argc, char** argv) {
    const int index = argc > 1 ? std::stoi(argv[1]) : 0;
    const auto device = dl::Device::cuda(index);
    // 重复训练可保留一部分池内空间；阶段结束后显式归还空闲资源。
    dl::cuda::set_memory_release_threshold(64ULL * 1024 * 1024, index);
    {
        const auto input = dl::ones({2, 3, 8, 8}, {.device = device});
        auto weight = dl::full({4, 3, 3, 3}, 0.01, {.device = device, .requires_grad = true});
        std::optional<dl::optim::AdamMoments> moments;
        dl::synchronize(device);
        dl::cuda::reset_peak_memory_stats(index);
        for (int step = 0; step < 4; ++step) {
            const auto output = dl::conv2d(input, weight, std::nullopt, {.padding = {1, 1}});
            const auto loss = dl::mean(output * output);
            auto gradients = dl::grad(loss, {weight});
            dl::optim::adamw_(weight, *gradients[0], moments);
        }
        dl::synchronize(device);
        print_memory("training state", index);
    }

    // 张量释放后，工作区与空闲池块一起归还；cuDNN 图计划仍可供下一阶段复用。
    dl::cuda::release_unused_memory(index);
    print_memory("released", index);
}

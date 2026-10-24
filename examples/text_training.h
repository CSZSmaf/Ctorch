#pragma once

#include "training.h"
#include "text_data.h"
#include "project_models.h"

#include <iterator>

namespace examples {

inline int train_text(int argc, char** argv, TextModelOptions config = {}) {
    if (argc < 3 || argc % 2 == 0) {
        std::cerr << "usage: " << argv[0] << " <local-text-file> <output-directory>"
                  << " [--device cpu|index] [--steps N] [--batch-size N] [--eval-every N]"
                  << " [--micro-batch-size N] [--clip-grad-norm value]"
                  << " [--precision f32|bf16|f16] [--attention math|cudnn]"
                  << " [--resume checkpoint] [--stop-after step]\n";
        return 1;
    }
    auto options = training_options(argc, argv, 8);
    std::ifstream source;
    source.exceptions(std::ios::failbit | std::ios::badbit);
    source.open(argv[1], std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
    // 先划分原始区间再构造序列，训练输入和目标都留在训练区间。
    const std::string_view bytes = text;
    const auto train_end = bytes.size() * 9 / 10;
    const auto validation_end = bytes.size() * 95 / 100;
    const TextDataset training{bytes.substr(0, train_end), config.length};
    const TextDataset validation_data{bytes.substr(train_end, validation_end - train_end), config.length};
    const TextDataset test_data{bytes.substr(validation_end), config.length};
    dl::Generator sampler(7), evaluation_sampler(0);
    dl::data::DataLoader source_loader(training.size(), training, {.batch_size = options.batch_size}, sampler);
    dl::data::PrefetchLoader loader(source_loader, options.device);
    dl::data::DataLoader validation(validation_data.size(), validation_data,
        {.batch_size = 32, .shuffle = false}, evaluation_sampler);
    dl::manual_seed(42);
    ByteTransformer model(config);
    model.to(options.device);
    dl::optim::AdamW optimizer(model.parameters(), {.learning_rate = 0.001});
    TrainingState state;
    auto progress = begin_training(options, model, optimizer, loader, validation, state,
        [&] { model.set_attention_backend(options.attention_backend); });
    const auto ignored_target = dl::full({}, -100, {.dtype = dl::DType::i64, .device = options.device});
    while (progress.step < options.steps) {
        if (progress.step >= options.stop_after) {
            save_training(options, model, optimizer, progress, loader, state);
            std::cout << "saved step " << progress.step << std::endl;
            return 0;
        }
        auto batch = loader.next();
        if (!batch) {
            ++progress.epoch;
            loader.reset();
            batch = loader.next();
        }
        // 每批有效目标数在 FP32 的精确整数范围内，跨批总数由 TrainingState 按 int64 累加。
        auto count = dl::sum(dl::ne(batch->target, ignored_target).astype(dl::DType::f32));
        const bool updated = train_batch(options, model, *batch, optimizer, state, std::move(count));
        if (updated) ++progress.step;
        if (updated && (progress.step % options.eval_every == 0 || progress.step == options.steps)) {
            record_validation(options, model, progress, state,
                evaluate(model, validation, options.device, options.precision, options.micro_batch_size));
            save_training(options, model, optimizer, progress, loader, state);
        }
    }
    dl::data::DataLoader test(test_data.size(), test_data, {.batch_size = 32, .shuffle = false}, evaluation_sampler);
    finish_training(options, model, test, state, progress, true);
    return 0;
}

} // namespace examples

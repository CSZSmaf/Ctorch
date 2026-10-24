#include "training.h"
#include "text_data.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>

namespace {

constexpr dl::Index sequence_length = 32;
constexpr dl::Index dimensions = 32;
constexpr dl::Index vocabulary = 256;

class ByteTransformer : public dl::nn::UnaryModule {
public:
    ByteTransformer()
        : positions_(register_buffer("positions", make_positions())),
          mask_(register_buffer("causal_mask", dl::causal_mask(sequence_length, sequence_length))) {
        tokens_ = add_module<dl::nn::Embedding>("tokens", vocabulary, dimensions);
        positions_embedding_ = add_module<dl::nn::Embedding>("positions_embedding", sequence_length, dimensions);
        encoder_ = add_module<dl::nn::TransformerEncoder>("encoder", 2, dimensions, 4, 4 * dimensions, 0.1);
        // 输入词表与输出分类器共享同一权重身份，优化器只更新这份参数一次。
        output_ = add_module<dl::nn::Linear>("output", tokens_->weight());
    }

    void set_attention_backend(dl::AttentionBackend backend) {
        attention_backend_ = backend;
        // 组合路径复用已登记的掩码，融合路径将因果关系直接交给库图。
        encoder_->set_attention_backend(backend, backend == dl::AttentionBackend::cudnn);
    }

    dl::Tensor forward(const dl::Tensor& input) override {
        auto hidden = (*tokens_)(input) + (*positions_embedding_)(bound_buffer(positions_));
        const auto mask = attention_backend_ == dl::AttentionBackend::math
            ? std::optional{bound_buffer(mask_)} : std::nullopt;
        return (*output_)(encoder_->forward(hidden, mask));
    }

private:
    static dl::Tensor make_positions() {
        std::vector<dl::Index> values(static_cast<std::size_t>(sequence_length));
        std::iota(values.begin(), values.end(), dl::Index{0});
        return dl::from_span<dl::Index>(values, {sequence_length});
    }

    dl::Tensor positions_, mask_;
    dl::AttentionBackend attention_backend_ = dl::AttentionBackend::math;
    std::shared_ptr<dl::nn::Embedding> tokens_, positions_embedding_;
    std::shared_ptr<dl::nn::TransformerEncoder> encoder_;
    std::shared_ptr<dl::nn::Linear> output_;
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc % 2 == 0) {
        std::cerr << "usage: ctorch_byte_transformer <local-text-file> <output-directory>"
                  << " [--device cpu|index] [--steps N] [--batch-size N] [--eval-every N]"
                  << " [--micro-batch-size N] [--clip-grad-norm value]"
                  << " [--precision f32|bf16|f16] [--attention math|cudnn]"
                  << " [--resume checkpoint] [--stop-after step]\n";
        return 1;
    }
    auto options = examples::training_options(argc, argv, 8);
    std::ifstream source;
    source.exceptions(std::ios::failbit | std::ios::badbit);
    source.open(argv[1], std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
    // 先划分原始区间再构造序列，训练输入和目标都留在训练区间。
    const std::string_view bytes = text;
    const auto train_end = bytes.size() * 9 / 10;
    const auto validation_end = bytes.size() * 95 / 100;
    const examples::TextDataset training{bytes.substr(0, train_end), sequence_length};
    const examples::TextDataset validation_data{bytes.substr(train_end, validation_end - train_end), sequence_length};
    const examples::TextDataset test_data{bytes.substr(validation_end), sequence_length};
    dl::Generator sampler(7), evaluation_sampler(0);
    dl::data::DataLoader source_loader(training.size(), training, {.batch_size = options.batch_size}, sampler);
    dl::data::PrefetchLoader loader(source_loader, options.device);
    dl::data::DataLoader validation(validation_data.size(), validation_data,
        {.batch_size = 32, .shuffle = false}, evaluation_sampler);
    dl::manual_seed(42);
    ByteTransformer model;
    model.to(options.device);
    dl::optim::AdamW optimizer(model.parameters(), {.learning_rate = 0.001});
    examples::TrainingState state;
    auto progress = examples::begin_training(options, model, optimizer, loader, validation, state,
        [&] { model.set_attention_backend(options.attention_backend); });
    const auto ignored_target = dl::full({}, -100, {.dtype = dl::DType::i64, .device = options.device});
    while (progress.step < options.steps) {
        if (progress.step >= options.stop_after) {
            examples::save_training(options, model, optimizer, progress, loader, state);
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
        const bool updated = examples::train_batch(options, model, *batch, optimizer, state, std::move(count));
        if (updated) ++progress.step;
        if (updated && (progress.step % options.eval_every == 0 || progress.step == options.steps)) {
            examples::record_validation(options, model, progress, state,
                examples::evaluate(model, validation, options.device, options.precision, options.micro_batch_size));
            examples::save_training(options, model, optimizer, progress, loader, state);
        }
    }
    dl::data::DataLoader test(test_data.size(), test_data, {.batch_size = 32, .shuffle = false}, evaluation_sampler);
    examples::finish_training(options, model, test, state, progress, true);
}

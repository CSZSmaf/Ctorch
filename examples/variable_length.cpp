#include <dl/dl.h>

#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

struct Sample { dl::Index label; std::vector<dl::Index> tokens; };

// 每行使用“整数标签、制表符、非空 UTF-8 文本”；按字节编码，零专用于填充。
std::vector<Sample> read_samples(const std::filesystem::path& path) {
    std::ifstream input;
    input.exceptions(std::ios::badbit);
    input.open(path);
    if (!input) throw std::runtime_error("cannot open text dataset");
    std::vector<Sample> samples;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        const auto tab = line.find('\t');
        Sample sample{std::stoll(line.substr(0, tab)), {}};
        for (unsigned char byte : line.substr(tab + 1)) sample.tokens.push_back(static_cast<dl::Index>(byte) + 1);
        samples.push_back(std::move(sample));
    }
    return samples;
}

dl::data::Collate collate(const std::vector<Sample>& samples) {
    return [&samples](std::span<const dl::Index> ids) {
        std::vector<dl::Tensor> sequences;
        std::vector<dl::Index> labels;
        for (auto id : ids) {
            const auto& sample = samples[id];
            sequences.push_back(dl::from_span<dl::Index>(sample.tokens, {static_cast<dl::Index>(sample.tokens.size())}));
            labels.push_back(sample.label);
        }
        auto padded = dl::data::pad_sequence(sequences);
        return dl::data::Batch{padded.values, dl::from_span<dl::Index>(labels, {static_cast<dl::Index>(ids.size())}),
            {padded.mask, padded.lengths}};
    };
}

class TextClassifier : public dl::nn::Module {
public:
    explicit TextClassifier(dl::Index classes) {
        embedding = add_module<dl::nn::Embedding>("embedding", 257, 32);
        classifier = add_module<dl::nn::Linear>("classifier", 32, classes);
    }
    dl::Tensor operator()(const dl::data::Batch& batch) {
        auto mask = batch.extra[0].astype(dl::DType::f32).unsqueeze(-1);
        auto features = dl::sum((*embedding)(batch.input) * mask, {1}) / dl::sum(mask, {1});
        return (*classifier)(features);
    }
private:
    std::shared_ptr<dl::nn::Embedding> embedding;
    std::shared_ptr<dl::nn::Linear> classifier;
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "usage: ctorch_variable_length <train.tsv> <validation.tsv> <classes> <output-directory> [cpu|cuda]\n";
        return 0;
    }
    const auto train = read_samples(argv[1]), validation = read_samples(argv[2]);
    const auto classes = std::stoll(argv[3]);
    const auto output = std::filesystem::path(argv[4]);
    std::filesystem::create_directories(output);
    const auto device = argc > 5 && std::string_view(argv[5]) == "cuda" ? dl::Device::cuda() : dl::Device::cpu();
    dl::manual_seed(12);
    dl::Generator sampling(13);
    dl::data::DataLoader source(static_cast<dl::Index>(train.size()), collate(train), {.batch_size = 16}, sampling);
    dl::data::DataLoader valid_source(static_cast<dl::Index>(validation.size()), collate(validation),
        {.batch_size = 16, .shuffle = false}, sampling);
    dl::data::PrefetchLoader batches(source, device, 2, true), valid_batches(valid_source, device, 2, true);
    dl::Shape lengths;
    for (const auto& sample : train) lengths.push_back(static_cast<dl::Index>(sample.tokens.size()));
    TextClassifier model(classes);
    model.to(device);
    dl::optim::AdamW optimizer(model.parameters(), {.learning_rate = .02});
    dl::training::EarlyStopping stopping({.patience = 3});
    for (dl::Index epoch = 0; epoch < 12; ++epoch) {
        model.train();
        batches.reset(dl::data::bucket_order(lengths, 16, sampling));
        while (auto batch = batches.next()) {
            optimizer.zero_grad();
            dl::cross_entropy(model(*batch), batch->target, dl::CrossEntropyOptions{.label_smoothing = .05}).backward();
            optimizer.step();
        }
        model.eval();
        valid_batches.reset();
        dl::training::ConfusionMatrix metrics(classes);
        double valid_loss = 0;
        {
            dl::NoGradGuard guard;
            while (auto batch = valid_batches.next()) {
                auto logits = model(*batch);
                valid_loss += dl::cross_entropy(logits, batch->target).item<float>() * batch->target.numel();
                metrics.update(dl::argmax(logits, 1), batch->target);
            }
        }
        valid_loss /= static_cast<double>(validation.size());
        if (epoch == 0 || valid_loss < stopping.best()) dl::io::save(output / "best.ct", model.state_dict());
        std::cout << "epoch=" << epoch + 1 << " validation_loss=" << valid_loss
                  << " accuracy=" << metrics.scores().accuracy << '\n';
        if (stopping.step(valid_loss)) break;
    }
}

#pragma once

#include <dl/dl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <variant>
#include <vector>

namespace examples {

/// 示例共用输出与运行选项；轮数、步数和批量大小均为正。
/// micro_batch_size、clip_grad_norm 非负，零分别表示整批计算和关闭裁剪。
struct TrainingOptions {
    dl::Device device = dl::Device::cpu();
    dl::DType precision = dl::DType::f32;
    dl::AttentionBackend attention_backend = dl::AttentionBackend::math;
    std::filesystem::path output, resume;
    dl::Index epochs = 5, steps = 200, batch_size = 64, eval_every = 50;
    dl::Index micro_batch_size = 0;
    double clip_grad_norm = 0.0;
    dl::Index stop_after = std::numeric_limits<dl::Index>::max();
};

inline TrainingOptions training_options(int argc, char** argv, dl::Index batch_size = 64) {
    TrainingOptions options;
    options.output = argv[2];
    options.batch_size = batch_size;
    for (int index = 3; index < argc; index += 2) {
        const std::string_view name = argv[index], value = argv[index + 1];
        if (name == "--device") options.device = value == "cpu"
            ? dl::Device::cpu() : dl::Device::cuda(std::stoi(std::string(value)));
        else if (name == "--resume") options.resume = value;
        else if (name == "--precision") options.precision = value == "bf16" ? dl::DType::bf16
            : value == "f16" ? dl::DType::f16 : dl::DType::f32;
        else if (name == "--attention") {
            if (value == "math") options.attention_backend = dl::AttentionBackend::math;
            else if (value == "cudnn") options.attention_backend = dl::AttentionBackend::cudnn;
            else throw std::invalid_argument("unknown attention backend: " + std::string(value));
        }
        else if (name == "--epochs") options.epochs = std::stoll(std::string(value));
        else if (name == "--steps") options.steps = std::stoll(std::string(value));
        else if (name == "--batch-size") options.batch_size = std::stoll(std::string(value));
        else if (name == "--micro-batch-size") options.micro_batch_size = std::stoll(std::string(value));
        else if (name == "--clip-grad-norm") options.clip_grad_norm = std::stod(std::string(value));
        else if (name == "--eval-every") options.eval_every = std::stoll(std::string(value));
        else if (name == "--stop-after") options.stop_after = std::stoll(std::string(value));
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    std::filesystem::create_directories(options.output);
    return options;
}

inline std::ofstream output_file(const std::filesystem::path& path, bool append = false) {
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(path, append ? std::ios::app : std::ios::out);
    output << std::setprecision(12);
    return output;
}

/// 评估临时切换整个模块树，结束后恢复各子模块原有的模式。
class Evaluation {
public:
    explicit Evaluation(dl::nn::Module& model) : model_(model), modes_(model.training_modes()) { model_.eval(); }
    ~Evaluation() { model_.load_training_modes(modes_); }
private:
    dl::nn::Module& model_;
    dl::nn::TrainingModes modes_;
    dl::NoGradGuard no_grad_;
};

/// 指标以有界窗口保留 FP32 标量，结算后使用 double / int64 汇总。
/// 设备计数是 FP32 可精确表示的整数，宿主计数保留原有 Index 类型。
struct Metrics {
    using Count = std::variant<dl::Index, dl::Tensor>;

    double loss_sum = 0;
    dl::Index count = 0, correct = 0;
    double loss() const { return loss_sum / static_cast<double>(count); }
    double accuracy() const { return static_cast<double>(correct) / static_cast<double>(count); }

    /// loss 是平均值或总和；训练提交平均值，评估提交总和与可选命中数。
    void record(const dl::Tensor& loss, Count count, dl::LossReduction reduction = dl::LossReduction::mean,
                std::optional<dl::Tensor> correct = {}) {
        if (auto* value = std::get_if<dl::Tensor>(&count)) *value = value->detach();
        if (correct) *correct = correct->detach();
        pending_.push_back({loss.detach(), std::move(count), std::move(correct), reduction});
        if (pending_.size() == 256) flush();
    }

    /// 整窗一次读回，损失按提交顺序累计；跨批计数保持整数精度。
    void flush() {
        if (pending_.empty()) return;
        std::vector<dl::Tensor> values;
        values.reserve(3 * pending_.size());
        for (const auto& batch : pending_) {
            values.push_back(batch.loss);
            if (const auto* count = std::get_if<dl::Tensor>(&batch.count)) values.push_back(*count);
            if (batch.correct) values.push_back(*batch.correct);
        }
        const auto host = dl::stack(values).to_vector<float>();
        std::size_t index = 0;
        for (const auto& batch : pending_) {
            const double loss = host[index++];
            const auto* host_count = std::get_if<dl::Index>(&batch.count);
            const auto valid = host_count ? *host_count : static_cast<dl::Index>(host[index++]);
            loss_sum += batch.reduction == dl::LossReduction::mean ? loss * static_cast<double>(valid) : loss;
            count += valid;
            if (batch.correct) correct += static_cast<dl::Index>(host[index++]);
        }
        pending_.clear();
    }

    void clear() {
        loss_sum = 0;
        count = correct = 0;
        pending_.clear();
    }

private:
    struct Batch {
        dl::Tensor loss;
        Count count;
        std::optional<dl::Tensor> correct;
        dl::LossReduction reduction;
    };
    std::vector<Batch> pending_;
};

/// 沿样本轴交付输入与标签的对应视图；limit 为零时直接交付整批。
template <class Function>
void for_each_micro_batch(const dl::data::Batch& batch, dl::Index limit, Function&& function) {
    const auto samples = batch.input.size(0);
    if (limit == 0 || limit >= samples) {
        std::invoke(function, batch);
        return;
    }
    for (dl::Index begin = 0; begin < samples; begin += limit) {
        const auto length = std::min(limit, samples - begin);
        std::invoke(function, dl::data::Batch{
            batch.input.narrow(0, begin, length), batch.target.narrow(0, begin, length)});
    }
}

/// 分类与语言模型都将最后一维作为类别；-100 的目标不计入分母。
inline Metrics evaluate(dl::nn::UnaryModule& model, dl::data::DataLoader& loader, dl::Device device,
                        dl::DType precision = dl::DType::f32, dl::Index micro_batch_size = 0) {
    Evaluation evaluation(model);
    dl::AutocastGuard autocast(device.type, precision, precision != dl::DType::f32);
    dl::data::PrefetchLoader batches(loader, device);
    batches.reset();
    Metrics metrics;
    const auto ignored = dl::full({}, -100, {.dtype = dl::DType::i64, .device = device});
    while (auto batch = batches.next()) {
        // 预取器交付完整设备批次，微批只建立视图；CUDA 上传可与当前批次计算衔接。
        for_each_micro_batch(*batch, micro_batch_size, [&](const dl::data::Batch& part) {
            auto output = model(part.input);
            auto logits = output.reshape({part.target.numel(), output.shape().back()});
            auto targets = part.target.reshape({-1});
            auto loss = dl::cross_entropy(logits, targets, dl::LossReduction::sum, -100);
            auto count = dl::sum(dl::ne(targets, ignored).astype(dl::DType::f32));
            // argmax 返回非负类别编号，-100 标签自然不计入命中数。
            auto correct = dl::sum(dl::eq(dl::argmax(logits, 1), targets).astype(dl::DType::f32));
            metrics.record(loss, std::move(count), dl::LossReduction::sum, std::move(correct));
        });
    }
    metrics.flush();
    return metrics;
}

inline dl::Tensor forward(dl::nn::UnaryModule& model, const dl::Tensor& input, const TrainingOptions& options) {
    dl::AutocastGuard autocast(options.device.type, options.precision, options.precision != dl::DType::f32);
    return model(input);
}

/// 最佳模型使用独立 CPU 存储，与后续更新及最新优化器状态分开保存。
struct TrainingState {
    Metrics metrics;
    double initial_loss = 0, best_loss = std::numeric_limits<double>::infinity();
    dl::Index best_step = 0, skipped_updates = 0;
    dl::StateDict best_model;
    dl::GradScaler scaler{{.enabled = false}};

    void consider(const dl::nn::Module& model, double loss, dl::Index step) {
        if (loss >= best_loss) return;
        best_loss = loss;
        best_step = step;
        best_model = model.state_dict();
        for (auto& [name, tensor] : best_model) {
            auto value = tensor.detach();
            tensor = value.device() == dl::Device::cpu() ? value.clone() : value.to(dl::Device::cpu());
        }
    }

    dl::StateDict extra() const {
        dl::StateDict result;
        for (const auto& [name, value] : std::array{
                 std::pair{"loss_sum", metrics.loss_sum}, std::pair{"initial_loss", initial_loss}, std::pair{"best_loss", best_loss}}) {
            result.emplace_back(name, dl::full({}, value, {.dtype = dl::DType::f64}));
        }
        for (const auto& [name, value] : std::array{std::pair{"count", metrics.count}, std::pair{"best_step", best_step},
                                                  std::pair{"skipped_updates", skipped_updates}}) {
            result.emplace_back(name, dl::from_span<dl::Index>(std::span(&value, 1), {}));
        }
        for (const auto& [name, tensor] : best_model) result.emplace_back("best/" + name, tensor);
        return result;
    }

    void restore(const dl::StateDict& extra) {
        metrics.clear();
        best_model.clear();
        for (const auto& [name, tensor] : extra) {
            if (name == "loss_sum") metrics.loss_sum = tensor.item<double>();
            else if (name == "initial_loss") initial_loss = tensor.item<double>();
            else if (name == "best_loss") best_loss = tensor.item<double>();
            else if (name == "count") metrics.count = tensor.item<dl::Index>();
            else if (name == "best_step") best_step = tensor.item<dl::Index>();
            else if (name == "skipped_updates") skipped_updates = tensor.item<dl::Index>();
            else if (name.starts_with("best/")) best_model.emplace_back(name.substr(5), tensor);
        }
    }
};

/// 一个 loader batch 是完整更新窗口，count 为其中有效样本或目标的正数。
/// 拆分时每个损失使用同一反向种子，逐块释放计算图，梯度累计到已有叶槽。
inline bool train_batch(const TrainingOptions& options, dl::nn::UnaryModule& model, const dl::data::Batch& batch,
    dl::optim::Optimizer& optimizer, TrainingState& state, Metrics::Count count) {
    const auto samples = batch.input.size(0);
    const bool split = options.micro_batch_size > 0 && options.micro_batch_size < samples;
    std::optional<dl::Tensor> denominator;
    if (split) {
        if (const auto* host_count = std::get_if<dl::Index>(&count))
            denominator = dl::full({}, static_cast<double>(*host_count), {.device = options.device});
        else denominator = std::get<dl::Tensor>(count);
    }
    const auto seed = denominator
        ? std::optional{dl::full({}, state.scaler.get_scale(), {.device = options.device}) / *denominator}
        : std::nullopt;
    std::optional<dl::Tensor> loss_total;
    optimizer.zero_grad();
    for_each_micro_batch(batch, options.micro_batch_size, [&](const dl::data::Batch& part) {
        auto target = part.target;
        auto logits = forward(model, part.input, options);
        if (logits.ndim() != 2) logits = logits.reshape({-1, logits.shape().back()});
        if (target.ndim() != 1) target = target.reshape({-1});
        auto loss = dl::cross_entropy(logits, target,
            split ? dl::LossReduction::sum : dl::LossReduction::mean, -100);
        if (seed) loss.backward(*seed);
        else state.scaler.scale(loss).backward();
        auto value = loss.detach();
        loss_total = loss_total ? *loss_total + value : std::move(value);
    });

    // 全部微批完成后只还原一次；裁剪作用于有效 batch 的平均梯度。
    const bool updated = state.scaler.unscale(optimizer);
    if (updated) {
        if (options.clip_grad_norm > 0.0) dl::optim::clip_grad_norm(optimizer.parameters(), options.clip_grad_norm);
        optimizer.step();
    }
    state.scaler.update(updated);
    if (!updated) ++state.skipped_updates;
    state.metrics.record(denominator ? *loss_total / *denominator : *loss_total, std::move(count));
    return updated;
}

inline void save_training(const TrainingOptions& options, const dl::nn::Module& model,
    const dl::optim::Optimizer& optimizer, dl::io::TrainingProgress progress,
    const dl::data::PrefetchLoader& loader, TrainingState& state) {
    state.metrics.flush();
    auto checkpoint = dl::io::capture_checkpoint(model, optimizer, progress, loader, {}, {}, &state.scaler);
    checkpoint.extra = state.extra();
    checkpoint.extra.emplace_back("compute_precision", dl::full({}, static_cast<int>(options.precision), {.dtype = dl::DType::i64}));
    checkpoint.extra.emplace_back("attention_backend", dl::full({}, static_cast<int>(options.attention_backend), {.dtype = dl::DType::i64}));
    checkpoint.extra.emplace_back("micro_batch_size", dl::from_span<dl::Index>(std::span(&options.micro_batch_size, 1), {}));
    checkpoint.extra.emplace_back("clip_grad_norm", dl::full({}, options.clip_grad_norm, {.dtype = dl::DType::f64}));
    dl::io::save_checkpoint(options.output / "latest.train.ct", checkpoint);
}

inline dl::io::TrainingProgress begin_training(TrainingOptions& options, dl::nn::UnaryModule& model,
    dl::optim::Optimizer& optimizer, dl::data::PrefetchLoader& loader,
    dl::data::DataLoader& validation, TrainingState& state, std::function<void()> configure_model = [] {}) {
    // summary 对应一次完成的训练；继续更新后由新的最终评价生成。
    std::filesystem::remove(options.output / "summary.json");
    if (!options.resume.empty()) {
        const auto checkpoint = dl::io::load_checkpoint(options.resume);
        for (const auto& [name, tensor] : checkpoint.extra) {
            if (name == "compute_precision") options.precision = static_cast<dl::DType>(tensor.item<dl::Index>());
            else if (name == "attention_backend")
                options.attention_backend = static_cast<dl::AttentionBackend>(tensor.item<dl::Index>());
            else if (name == "micro_batch_size") options.micro_batch_size = tensor.item<dl::Index>();
            else if (name == "clip_grad_norm") options.clip_grad_norm = tensor.item<double>();
        }
        // 模型在状态恢复前取得快照中的计算配置，后续训练和评估沿用同一调用路径。
        configure_model();
        state.scaler = dl::GradScaler({.enabled = options.precision == dl::DType::f16});
        const auto progress = dl::io::restore_checkpoint(checkpoint, model, optimizer, loader, {}, {}, &state.scaler);
        state.restore(checkpoint.extra);
        return progress;
    }
    configure_model();
    state.scaler = dl::GradScaler({.enabled = options.precision == dl::DType::f16});
    for (const auto* name : {"metrics.csv", "best.ct", "last.ct"}) std::filesystem::remove(options.output / name);
    dl::io::save(options.output / "initial.ct", model.state_dict());
    state.initial_loss = evaluate(model, validation, options.device, options.precision, options.micro_batch_size).loss();
    state.consider(model, state.initial_loss, 0);
    std::cout << "initial validation loss " << state.initial_loss << std::endl;
    return {};
}

inline void record_validation(const TrainingOptions& options, const dl::nn::Module& model,
    dl::io::TrainingProgress progress, TrainingState& state, const Metrics& validation) {
    state.metrics.flush();
    const auto path = options.output / "metrics.csv";
    const bool existing = std::filesystem::exists(path);
    auto output = output_file(path, true);
    if (!existing) output << "epoch,step,train_loss,validation_loss,validation_accuracy\n";
    output << progress.epoch << ',' << progress.step << ',' << state.metrics.loss() << ','
           << validation.loss() << ',' << validation.accuracy() << '\n';
    std::cout << "step " << progress.step << ", train loss " << state.metrics.loss()
              << ", validation loss " << validation.loss() << ", accuracy " << validation.accuracy() << std::endl;
    state.consider(model, validation.loss(), progress.step);
    state.metrics.clear();
}

/// 最新快照已经保存后，再加载最佳权重做一次测试集评价。
inline void finish_training(const TrainingOptions& options, dl::nn::UnaryModule& model,
    dl::data::DataLoader& test_loader, const TrainingState& state, dl::io::TrainingProgress progress,
    bool language_model = false) {
    dl::io::save(options.output / "last.ct", model.state_dict());
    dl::io::save(options.output / "best.ct", state.best_model);
    model.load_state_dict(state.best_model);
    const auto test = evaluate(model, test_loader, options.device, options.precision, options.micro_batch_size);
    auto summary = output_file(options.output / "summary.json");
    summary << "{\n  \"step\": " << progress.step << ",\n  \"epoch\": " << progress.epoch
            << ",\n  \"best_step\": " << state.best_step << ",\n  \"initial_validation_loss\": " << state.initial_loss
            << ",\n  \"best_validation_loss\": " << state.best_loss << ",\n  \"test_count\": " << test.count
            << ",\n  \"test_loss\": " << test.loss() << ",\n  \"test_accuracy\": " << test.accuracy();
    if (language_model) {
        summary << ",\n  \"test_perplexity\": " << std::exp(test.loss());
        summary << ",\n  \"attention_backend\": \""
                << (options.attention_backend == dl::AttentionBackend::cudnn ? "cudnn" : "math") << '"';
    }
    summary << ",\n  \"precision\": \"" << (options.precision == dl::DType::bf16 ? "bf16" : options.precision == dl::DType::f16 ? "f16" : "f32") << '"';
    summary << ",\n  \"micro_batch_size\": " << options.micro_batch_size
            << ",\n  \"clip_grad_norm\": " << options.clip_grad_norm;
    summary << ",\n  \"skipped_updates\": " << state.skipped_updates << "\n}\n";
    std::cout << "best step " << state.best_step << ", test loss " << test.loss()
              << ", accuracy " << test.accuracy();
    if (language_model) std::cout << ", perplexity " << std::exp(test.loss());
    std::cout << std::endl;
}

} // namespace examples

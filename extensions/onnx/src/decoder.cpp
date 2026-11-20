#include "dl/onnx.h"
#include "dl/decoder.h"
#include "core/tensor_impl.h"
#include "model_writer.h"

#include <array>
#include <stdexcept>
#include <unordered_map>
#include <variant>

namespace dl::onnx {
namespace {

using Dim = std::variant<Index, std::string_view>;

struct DecoderWriter : detail::ModelWriter {
    explicit DecoderWriter(const nn::TransformerDecoder& decoder) {
        graph.data(2, "CtorchDecoder");
        for (auto& entry : decoder.state_dict()) state.insert(std::move(entry));
    }
    std::unordered_map<std::string, Tensor> state;
    std::unordered_map<const void*, std::string> weights;

    void port(unsigned field, const std::string& name, DType dtype, std::initializer_list<Dim> dimensions) {
        detail::Message shape;
        for (const auto& extent : dimensions) {
            detail::Message dimension;
            if (const auto* n = std::get_if<Index>(&extent)) dimension.integer(1, *n);
            else dimension.data(2, std::get<std::string_view>(extent));
            shape.message(1, dimension);
        }
        graph.message(field, detail::value_info(name, dtype, shape));
    }
    std::string weight(const Tensor& tensor) {
        // 共享词嵌入和输出权重在文件中引用同一个 initializer。
        auto [entry, inserted] = weights.try_emplace(dl::detail::TensorAccess::identity(tensor));
        if (inserted) entry->second = constant(tensor.detach().to(Device::cpu()).astype(DType::f32));
        return entry->second;
    }
    std::string linear(const std::string& x, const Tensor& matrix) {
        auto transposed = intermediate("Transpose", {weight(matrix)}, {{"perm", Shape{1, 0}}});
        return intermediate("MatMul", {x, transposed});
    }
    std::string norm(const std::string& x, const std::string& name) {
        return intermediate("RMSNormalization", {x, weight(state.at(name))},
                            {{"epsilon", nn::TransformerDecoderLayer::norm_epsilon}});
    }
    std::string positions(const std::string& input) {
        const auto one = constant(full({}, 1, {.dtype = DType::i64}));
        const auto two = constant(full({}, 2, {.dtype = DType::i64}));
        auto batch_tokens = intermediate("Shape", {input}, {{"end", Index{2}}});
        auto count = intermediate("Gather", {batch_tokens, one});
        auto past = intermediate("Gather", {intermediate("Shape", {"past_key_0"}), two});
        auto range = intermediate("Range", {past, intermediate("Add", {past, count}), one});
        return intermediate("Expand", {range, batch_tokens});
    }
    std::string layer(Index index, const std::string& x, const std::string& positions) {
        const auto suffix = std::to_string(index), layer = "layer" + suffix + ".";
        const auto attention = layer + "attention.", ff = layer + "feedforward.";
        const auto& cosine = state.at(attention + "rotary.cosine");
        const auto width = state.at(attention + "query.weight").size(0);
        const auto head_size = 2 * cosine.size(1);
        const auto kv_heads = state.at(attention + "key.weight").size(0) / head_size;
        const auto past_k = "past_key_" + suffix, past_v = "past_value_" + suffix;
        const auto present_k = "present_key_" + suffix, present_v = "present_value_" + suffix;
        for (const auto& name : {past_k, past_v})
            port(11, name, DType::f32, {"batch", kv_heads, "past", head_size});
        for (const auto& name : {present_k, present_v})
            port(12, name, DType::f32, {"batch", kv_heads, "total", head_size});

        const auto normalized = norm(x, layer + "attention_norm.weight");
        const std::vector<plan::Attribute> swap_heads{{"perm", Shape{0, 2, 1, 3}}};
        auto project = [&](const char* name, Index heads) {
            auto projected = linear(normalized, state.at(attention + name + ".weight"));
            auto split = intermediate("Reshape", {projected, indices({0, 0, heads, head_size})});
            return intermediate("Transpose", {split}, swap_heads);
        };
        auto rotate = [&](const std::string& input) {
            return intermediate("RotaryEmbedding", {input, weight(cosine), weight(state.at(attention + "rotary.sine")), positions},
                                {{"interleaved", Index{1}}});
        };
        auto q = rotate(project("query", width / head_size)), k = rotate(project("key", kv_heads));
        auto v = project("value", kv_heads), attended = fresh();
        const std::array<plan::Attribute, 1> causal{{{"is_causal", Index{1}}}};
        node("Attention", {q, k, v, "", past_k, past_v}, {attended, present_k, present_v}, causal);
        auto merged = intermediate("Reshape", {intermediate("Transpose", {attended}, swap_heads), indices({0, 0, width})});
        auto residual = intermediate("Add", {x, linear(merged, state.at(attention + "output.weight"))});
        auto features = norm(residual, layer + "feedforward_norm.weight");
        auto gate = linear(features, state.at(ff + "gate.weight"));
        auto silu = intermediate("Mul", {gate, intermediate("Sigmoid", {gate})});
        auto hidden = intermediate("Mul", {silu, linear(features, state.at(ff + "up.weight"))});
        return intermediate("Add", {residual, linear(hidden, state.at(ff + "down.weight"))});
    }
};

void write_decoder(const std::filesystem::path& path, const nn::TransformerDecoder& decoder,
                   const nn::Embedding* embedding, const nn::Linear* output) {
    DecoderWriter writer(decoder);
    const auto width = writer.state.at("normalization.weight").size(0);
    const std::string input = embedding ? "input_ids" : "features";
    const std::string result = output ? "logits" : "hidden_states";
    if (embedding) {
        if (embedding->weight().size(1) != width || output->weight().size(1) != width)
            throw std::invalid_argument("decoder input and output projections must match its hidden width");
        writer.port(11, input, DType::i64, {"batch", "tokens"});
    } else writer.port(11, input, DType::f32, {"batch", "tokens", width});
    writer.port(12, result, DType::f32, {"batch", "tokens", output ? output->weight().size(0) : width});
    auto positions = writer.positions(input);
    auto x = embedding ? writer.intermediate("Gather", {writer.weight(embedding->weight()), input}) : input;
    for (Index i = 0; i < decoder.layer_count(); ++i) x = writer.layer(i, x, positions);
    x = writer.norm(x, "normalization.weight");
    if (output) {
        x = writer.linear(x, output->weight());
        if (output->bias()) x = writer.intermediate("Add", {x, writer.weight(*output->bias())});
    }
    writer.emit("Identity", {x}, result);
    writer.write(path, 12, 24);
}

} // namespace

void save_decoder(const std::filesystem::path& path, const nn::TransformerDecoder& decoder) {
    write_decoder(path, decoder, nullptr, nullptr);
}
void save_decoder(const std::filesystem::path& path, const nn::Embedding& embedding,
                  const nn::TransformerDecoder& decoder, const nn::Linear& output) {
    write_decoder(path, decoder, &embedding, &output);
}

} // namespace dl::onnx

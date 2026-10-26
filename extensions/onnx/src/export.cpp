#include "dl/onnx.h"
#include "protobuf.h"

#include <fstream>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace dl::onnx {
namespace {

using detail::Message;
using plan::Attribute;
using plan::NodeInfo;
using plan::Value;

template <class T> T property(const NodeInfo& node, std::string_view name) {
    for (const auto& [key, value] : node.operation.attributes) if (key == name) return std::get<T>(value);
    throw std::invalid_argument("missing operator attribute: " + std::string(name));
}

struct Exporter {
    plan::GraphInfo info;
    Message graph;
    std::vector<std::string> names;
    std::unordered_set<std::string> used;
    std::size_t temporary = 0;

    std::string fresh() {
        std::string name;
        do { name = "value_" + std::to_string(temporary++); } while (!used.insert(name).second);
        return name;
    }
    std::string constant(const Tensor& value) {
        auto name = fresh();
        graph.message(5, detail::tensor_data(name, value));
        return name;
    }
    std::string indices(const Shape& values) {
        return constant(from_span<Index>(values, {static_cast<Index>(values.size())}));
    }
    void emit(std::string_view operation, const std::vector<std::string>& inputs, const std::string& output,
              std::span<const Attribute> attributes = {}) {
        Message node;
        for (const auto& input : inputs) node.data(1, input);
        node.data(2, output); node.data(4, operation);
        for (const auto& value : attributes) node.message(5, detail::attribute(value));
        graph.message(1, node);
    }
    std::string intermediate(std::string_view operation, const std::vector<std::string>& inputs,
                             std::vector<Attribute> attributes = {}) {
        auto name = fresh();
        emit(operation, inputs, name, attributes);
        return name;
    }

    void lower(const NodeInfo& node) {
        const auto& kind = node.operation.kind;
        if (kind.empty() || node.outputs.size() != 1) throw std::invalid_argument("unsupported ONNX operation: " + node.name);
        std::vector<std::string> inputs;
        for (auto value : node.inputs) inputs.push_back(names[value.index]);
        const auto& output = names[node.outputs[0].index];
        const auto& spec = info.values[node.inputs[0].index];
        const auto& result = info.values[node.outputs[0].index];
        const auto rank = static_cast<Index>(spec.shape.size());

        if (kind == "linear") {
            auto weight = intermediate("Transpose", {inputs[1]}, {{"perm", Shape{1, 0}}});
            if (inputs.size() == 2) emit("MatMul", {inputs[0], weight}, output);
            else emit("Add", {intermediate("MatMul", {inputs[0], weight}), inputs[2]}, output);
        } else if (kind == "reshape") {
            const std::vector<Attribute> attributes{{"allowzero", Index{1}}};
            emit("Reshape", {inputs[0], indices(result.shape)}, output, attributes);
        } else if (kind == "transpose") {
            auto axes = property<Shape>(node, "axes");
            Shape permutation(spec.shape.size());
            std::iota(permutation.begin(), permutation.end(), Index{0});
            std::swap(permutation[(axes[0] + rank) % rank], permutation[(axes[1] + rank) % rank]);
            const std::vector<Attribute> attributes{{"perm", std::move(permutation)}};
            emit("Transpose", inputs, output, attributes);
        } else if (kind == "add_relu") {
            emit("Relu", {intermediate("Add", inputs)}, output);
        } else if (kind == "silu") {
            emit("Mul", {inputs[0], intermediate("Sigmoid", inputs)}, output);
        } else if (kind == "layer_norm") {
            const auto axis = (property<Index>(node, "axis") + rank) % rank;
            Shape axes(static_cast<std::size_t>(rank - axis));
            std::iota(axes.begin(), axes.end(), axis);
            const auto axis_input = indices(axes);
            auto mean = intermediate("ReduceMean", {inputs[0], axis_input});
            auto centered = intermediate("Sub", {inputs[0], mean});
            auto variance = intermediate("ReduceMean", {intermediate("Mul", {centered, centered}), axis_input});
            auto epsilon = constant(full({}, property<double>(node, "epsilon"), {.dtype = spec.dtype}));
            auto denominator = intermediate("Sqrt", {intermediate("Add", {variance, epsilon})});
            auto normalized = intermediate("Div", {centered, denominator});
            if (inputs.size() > 1) normalized = intermediate("Mul", {normalized, inputs[1]});
            if (inputs.size() > 2) emit("Add", {normalized, inputs[2]}, output);
            else emit("Identity", {normalized}, output);
        } else if (kind == "batch_norm") {
            if (property<Index>(node, "training") || !property<Index>(node, "track"))
                throw std::invalid_argument("ONNX batch_norm requires evaluation with running statistics");
            const bool affine = property<Index>(node, "affine") != 0;
            auto weight = affine ? inputs[3] : constant(ones({spec.shape[1]}, {.dtype = spec.dtype}));
            auto bias = affine ? inputs[4] : constant(zeros({spec.shape[1]}, {.dtype = spec.dtype}));
            const std::vector<Attribute> attributes{{"epsilon", property<double>(node, "epsilon")}};
            emit("BatchNormalization", {inputs[0], weight, bias, inputs[1], inputs[2]}, output, attributes);
        } else {
            static const std::unordered_map<std::string, std::string> operators{
                {"add", "Add"}, {"sub", "Sub"}, {"mul", "Mul"}, {"div", "Div"},
                {"relu", "Relu"}, {"sigmoid", "Sigmoid"}, {"tanh", "Tanh"},
                {"matmul", "MatMul"}, {"softmax", "Softmax"}, {"identity", "Identity"},
                {"conv", "Conv"}, {"max_pool", "MaxPool"}, {"avg_pool", "AveragePool"},
                {"global_avg_pool", "GlobalAveragePool"}};
            auto found = operators.find(kind);
            if (found == operators.end()) throw std::invalid_argument("unsupported ONNX operation: " + node.name);
            const bool convolution = kind == "conv";
            const bool pooling = kind == "max_pool" || kind == "avg_pool";
            const bool global = kind == "global_avg_pool";
            if (convolution || pooling || global) {
                const auto spatial = convolution ? info.values[node.inputs[1].index].shape.size() - 2
                    : pooling ? property<Shape>(node, "kernel_shape").size()
                              : static_cast<std::size_t>(property<Index>(node, "spatial_rank"));
                if (spec.shape.size() != spatial + 2)
                    throw std::invalid_argument("ONNX spatial operations require a batch dimension: " + node.name);
            }
            emit(found->second, inputs, output, global ? std::span<const Attribute>{} : node.operation.attributes);
        }
    }
};

} // namespace

void save(const std::filesystem::path& path, const plan::Graph& source,
          std::span<const Value> outputs, std::span<const Binding> weights) {
    Exporter writer{source.describe(), {}, {}, {}, 0};
    auto& info = writer.info;
    std::vector<bool> needed(info.values.size()), keep(info.nodes.size());
    for (auto value : outputs) needed[value.index] = true;
    for (std::size_t i = info.nodes.size(); i-- > 0;) {
        const auto& node = info.nodes[i];
        if (node.effect != plan::Effect::none) throw std::invalid_argument("stateful ONNX operation: " + node.name);
        for (auto value : node.outputs) keep[i] = keep[i] || needed[value.index];
        if (keep[i]) for (auto value : node.inputs) needed[value.index] = true;
    }
    for (std::size_t i = 0; i < needed.size(); ++i) {
        if (needed[i] && info.values[i].dtype != DType::f32 && info.values[i].dtype != DType::f64)
            throw std::invalid_argument("ONNX export supports FP32 and FP64 graphs");
    }
    std::unordered_map<std::size_t, Tensor> frozen;
    for (const auto& binding : weights) {
        const auto& spec = info.values[binding.input.index];
        if (spec.shape != binding.value.shape() || spec.dtype != binding.value.dtype())
            throw std::invalid_argument("ONNX weight shape and dtype must match the graph input");
        const auto input = std::find_if(info.inputs.begin(), info.inputs.end(), [&](const auto& entry) { return entry.first == binding.input; });
        if (input == info.inputs.end()) throw std::invalid_argument("ONNX weights bind graph inputs");
        if (!frozen.emplace(binding.input.index, binding.value).second) throw std::invalid_argument("duplicate ONNX weight binding");
    }
    // 输入名由调用者指定，内部值采用连续编号。
    writer.names.resize(info.values.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) writer.used.insert("output_" + std::to_string(i));
    for (const auto& [value, name] : info.inputs) if (needed[value.index] && !name.empty()) {
        if (!writer.used.insert(name).second) throw std::invalid_argument("duplicate or reserved ONNX input name: " + name);
        writer.names[value.index] = name;
    }
    for (auto& name : writer.names) if (name.empty()) name = writer.fresh();
    writer.graph.data(2, "Ctorch");
    for (const auto& [value, name] : info.inputs) if (needed[value.index]) {
        auto constant = frozen.find(value.index);
        if (constant != frozen.end()) writer.graph.message(5, detail::tensor_data(writer.names[value.index], constant->second));
        else writer.graph.message(11, detail::value_info(writer.names[value.index], info.values[value.index]));
    }
    for (std::size_t i = 0; i < info.nodes.size(); ++i) if (keep[i]) writer.lower(info.nodes[i]);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto name = "output_" + std::to_string(i);
        writer.emit("Identity", {writer.names[outputs[i].index]}, name);
        writer.graph.message(12, detail::value_info(name, info.values[outputs[i].index]));
    }
    Message model, opset;
    model.integer(1, 8); model.data(2, "Ctorch"); model.message(7, writer.graph);
    opset.integer(2, 18); model.message(8, opset);
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(path, std::ios::binary | std::ios::trunc);
    file.write(model.bytes.data(), static_cast<std::streamsize>(model.bytes.size()));
    file.close();
}

} // namespace dl::onnx

#pragma once

#include "protobuf.h"

#include <filesystem>
#include <fstream>
#include <unordered_set>

namespace dl::onnx::detail {

struct ModelWriter {
    Message graph;
    std::unordered_set<std::string> used;
    std::size_t temporary = 0;

    std::string fresh() {
        std::string name;
        do { name = "value_" + std::to_string(temporary++); } while (!used.insert(name).second);
        return name;
    }
    std::string constant(const Tensor& value) {
        auto name = fresh();
        graph.message(5, tensor_data(name, value));
        return name;
    }
    std::string indices(const Shape& values) {
        return constant(from_span<Index>(values, {static_cast<Index>(values.size())}));
    }
    void node(std::string_view operation, const std::vector<std::string>& inputs,
              std::initializer_list<std::string> outputs, std::span<const plan::Attribute> attributes = {}) {
        Message op;
        for (const auto& input : inputs) op.data(1, input);
        for (const auto& output : outputs) op.data(2, output);
        op.data(4, operation);
        for (const auto& value : attributes) op.message(5, attribute(value));
        graph.message(1, op);
    }
    void emit(std::string_view operation, const std::vector<std::string>& inputs, const std::string& output,
              std::span<const plan::Attribute> attributes = {}) {
        node(operation, inputs, {output}, attributes);
    }
    std::string intermediate(std::string_view operation, const std::vector<std::string>& inputs,
                             std::vector<plan::Attribute> attributes = {}) {
        auto name = fresh();
        emit(operation, inputs, name, attributes);
        return name;
    }
    void write(const std::filesystem::path& path, Index ir, Index opset_version) const {
        Message model, opset;
        model.integer(1, ir); model.data(2, "Ctorch"); model.message(7, graph);
        opset.integer(2, opset_version); model.message(8, opset);
        std::ofstream file;
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(path, std::ios::binary | std::ios::trunc);
        file.write(model.bytes.data(), static_cast<std::streamsize>(model.bytes.size()));
        file.close();
    }
};

} // namespace dl::onnx::detail

#pragma once

#include "dl/plan.h"

#include <bit>
#include <algorithm>
#include <string>
#include <string_view>
#include <type_traits>

namespace dl::onnx::detail {

// ONNX 使用 Protobuf 的整数、定长浮点和长度前缀字段；此写入器只负责模型导出。
// 字段定义见 https://github.com/onnx/onnx/blob/main/onnx/onnx.proto。
struct Message {
    std::string bytes;
    void varint(std::uint64_t value) {
        while (value >= 128) { bytes.push_back(static_cast<char>((value & 127) | 128)); value >>= 7; }
        bytes.push_back(static_cast<char>(value));
    }
    void integer(unsigned field, std::uint64_t value) { varint(field << 3); varint(value); }
    void data(unsigned field, std::string_view value) {
        varint((field << 3) | 2); varint(value.size()); bytes.append(value);
    }
    void message(unsigned field, const Message& value) { data(field, value.bytes); }
    void floating(unsigned field, float value) {
        varint((field << 3) | 5);
        auto bits = std::bit_cast<std::uint32_t>(value);
        for (int i = 0; i < 4; ++i) { bytes.push_back(static_cast<char>(bits & 255)); bits >>= 8; }
    }
};

inline Index type_code(DType dtype) {
    constexpr Index codes[]{1, 11, 7, 9, 10, 16};
    return codes[static_cast<std::size_t>(dtype)];
}

inline Message value_info(std::string_view name, const TensorSpec& spec) {
    Message shape, tensor, type, result;
    for (auto extent : spec.shape) { Message dimension; dimension.integer(1, extent); shape.message(1, dimension); }
    tensor.integer(1, type_code(spec.dtype)); tensor.message(2, shape);
    type.message(1, tensor);
    result.data(1, name); result.message(2, type);
    return result;
}

inline Message tensor_data(std::string_view name, const Tensor& tensor) {
    auto dense = tensor.detach().to(Device::cpu()).contiguous();
    const auto size = static_cast<std::size_t>(dense.numel()) * element_size(dense.dtype());
    Message result;
    for (auto extent : dense.shape()) result.integer(1, extent);
    result.integer(2, type_code(dense.dtype())); result.data(8, name);
    std::string bytes;
    if (size != 0) bytes.assign(reinterpret_cast<const char*>(dense.data<std::byte>()), size);
    if constexpr (std::endian::native == std::endian::big) {
        const auto width = element_size(dense.dtype());
        for (std::size_t i = 0; i < size; i += width) std::reverse(bytes.begin() + i, bytes.begin() + i + width);
    }
    result.data(9, bytes);
    return result;
}

inline Message attribute(const plan::Attribute& attribute) {
    Message result;
    result.data(1, attribute.first);
    std::visit([&]<class T>(const T& value) {
        if constexpr (std::is_same_v<T, Index>) { result.integer(20, 2); result.integer(3, value); }
        else if constexpr (std::is_same_v<T, double>) { result.integer(20, 1); result.floating(2, static_cast<float>(value)); }
        else { result.integer(20, 7); for (auto item : value) result.integer(8, item); }
    }, attribute.second);
    return result;
}

} // namespace dl::onnx::detail

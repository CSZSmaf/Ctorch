#pragma once

#include "dl/nn.h"

#include <array>
#include <bit>
#include <istream>
#include <ostream>
#include <string_view>

namespace dl::io::archive {

/// 整数和浮点位模式采用小端顺序，张量与训练状态复用同一组字节编码。
template <class UInt> void write_integer(std::ostream& output, UInt value) {
    std::array<char, sizeof(UInt)> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<char>((value >> (8 * index)) & 0xff);
    }
    output.write(bytes.data(), bytes.size());
}

template <class UInt> UInt read_integer(std::istream& input) {
    std::array<unsigned char, sizeof(UInt)> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    UInt value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        value |= static_cast<UInt>(bytes[index]) << (8 * index);
    }
    return value;
}

inline void write_double(std::ostream& output, double value) {
    write_integer(output, std::bit_cast<std::uint64_t>(value));
}
inline double read_double(std::istream& input) {
    return std::bit_cast<double>(read_integer<std::uint64_t>(input));
}

void read_header(std::istream& input, const std::array<char, 8>& expected);
void write_text(std::ostream& output, std::string_view text);
std::string read_text(std::istream& input);
void write_tensor(std::ostream& output, const Tensor& tensor);
Tensor read_tensor(std::istream& input);
void write_state(std::ostream& output, std::span<const NamedTensor> state);
StateDict read_state(std::istream& input);

} // namespace dl::io::archive

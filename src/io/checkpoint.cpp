#include "dl/io.h"

#include "core/tensor_impl.h"
#include "ops/numeric.h"
#include "io/archive.h"
#include "io/output_file.h"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <stdexcept>

namespace dl::io {
namespace {

constexpr std::array<char, 8> format{'C', 'T', 'O', 'R', 'C', 'H', 0, 1};

void swap_scalar_bytes(detail::TensorData& data) {
    if constexpr (std::endian::native == std::endian::big) {
        const auto width = element_size(data.dtype);
        auto* bytes = static_cast<std::byte*>(data.mutable_address());
        for (Index index = 0; index < data.layout.numel(); ++index) {
            std::reverse(bytes + index * width, bytes + (index + 1) * width);
        }
    }
}

} // namespace

namespace archive {

void read_header(std::istream& input, const std::array<char, 8>& expected) {
    std::array<char, 8> header{};
    input.read(header.data(), header.size());
    if (header != expected) {
        throw std::runtime_error("unsupported Ctorch checkpoint format");
    }
}

void write_text(std::ostream& output, std::string_view text) {
    write_integer(output, static_cast<std::uint64_t>(text.size()));
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string read_text(std::istream& input) {
    std::string text(static_cast<std::size_t>(read_integer<std::uint64_t>(input)), '\0');
    input.read(text.data(), static_cast<std::streamsize>(text.size()));
    return text;
}

void write_tensor(std::ostream& output, const Tensor& tensor) {
    write_integer(output, static_cast<std::uint8_t>(tensor.dtype()));
    write_integer(output, static_cast<std::uint64_t>(tensor.ndim()));
    for (auto dimension : tensor.shape()) {
        write_integer(output, static_cast<std::uint64_t>(dimension));
    }
    auto dense = detail::numeric::to(detail::data_of(tensor), Device::cpu());
    if (std::endian::native == std::endian::big || !dense.layout.is_contiguous()) {
        dense = detail::numeric::clone(dense);
    }
    swap_scalar_bytes(dense);
    const auto bytes = static_cast<std::streamsize>(dense.layout.numel() * element_size(dense.dtype));
    output.write(static_cast<const char*>(dense.address()), bytes);
}

Tensor read_tensor(std::istream& input) {
    const auto dtype = static_cast<DType>(read_integer<std::uint8_t>(input));
    Shape shape(static_cast<std::size_t>(read_integer<std::uint64_t>(input)));
    for (auto& dimension : shape) {
        dimension = static_cast<Index>(read_integer<std::uint64_t>(input));
    }
    auto data = detail::empty_data(std::move(shape), dtype);
    input.read(static_cast<char*>(data.mutable_address()), static_cast<std::streamsize>(data.storage->bytes));
    swap_scalar_bytes(data);
    return detail::make_tensor(std::move(data));
}

void write_state(std::ostream& output, std::span<const NamedTensor> state) {
    write_integer(output, static_cast<std::uint64_t>(state.size()));
    for (const auto& [name, tensor] : state) {
        write_text(output, name);
        write_tensor(output, tensor);
    }
}

StateDict read_state(std::istream& input) {
    const auto count = read_integer<std::uint64_t>(input);
    StateDict result;
    result.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t entry = 0; entry < count; ++entry) {
        auto name = read_text(input);
        result.emplace_back(std::move(name), read_tensor(input));
    }
    return result;
}

} // namespace archive

void save(const std::filesystem::path& path, std::span<const NamedTensor> state) {
    archive::OutputFile file(path);
    auto& output = file.stream();
    output.write(format.data(), format.size());
    archive::write_state(output, state);
    file.commit();
}

StateDict load(const std::filesystem::path& path) {
    std::ifstream input;
    input.exceptions(std::ios::failbit | std::ios::badbit);
    input.open(path, std::ios::binary);
    archive::read_header(input, format);
    return archive::read_state(input);
}

} // namespace dl::io

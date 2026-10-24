#include "dl/npu.h"

#include <openvino/openvino.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <stdexcept>
#include <utility>

namespace dl::npu {
namespace {

// 表的顺序与 DType 一致，宿主存储在交换边界保持原有精度和位表示。
const std::array<ov::element::Type, 6> element_types{
    ov::element::f32, ov::element::f64, ov::element::i64,
    ov::element::boolean, ov::element::f16, ov::element::bf16
};

ov::element::Type element_type(DType dtype) {
    return element_types[static_cast<std::size_t>(dtype)];
}

DType dtype_of_port(const ov::element::Type& type) {
    const auto found = std::find(element_types.begin(), element_types.end(), type);
    if (found == element_types.end()) {
        throw std::runtime_error(std::string("NPU tensor exchange does not support ") + type.get_type_name());
    }
    return static_cast<DType>(found - element_types.begin());
}

ov::Shape runtime_shape(const Shape& shape) { return {shape.begin(), shape.end()}; }
Shape tensor_shape(const ov::Shape& shape) { return {shape.begin(), shape.end()}; }

std::vector<Port> describe(const std::vector<ov::Output<const ov::Node>>& ports) {
    std::vector<Port> result;
    result.reserve(ports.size());
    for (const auto& port : ports) {
        const auto& names = port.get_names();
        auto name = names.empty() ? std::string{} : *std::min_element(names.begin(), names.end());
        result.push_back({std::move(name), dtype_of_port(port.get_element_type()), tensor_shape(port.get_shape())});
    }
    return result;
}

Tensor copy_output(ov::Tensor source) {
    // 设备插件可为宿主输出提供带间隔的布局，交换时统一为 Ctorch 连续张量。
    if (!source.is_continuous()) {
        ov::Tensor dense(source.get_element_type(), source.get_shape());
        source.copy_to(dense);
        source = std::move(dense);
    }
    return dl::detail::from_host(std::as_const(source).data(), tensor_shape(source.get_shape()),
                                 dtype_of_port(source.get_element_type()), false, Device::cpu());
}

ov::AnyMap properties(const CompileOptions& options) {
    const auto mode = options.performance == Performance::latency
        ? ov::hint::PerformanceMode::LATENCY : ov::hint::PerformanceMode::THROUGHPUT;
    ov::AnyMap result{ov::hint::performance_mode(mode)};
    if (!options.cache_directory.empty()) {
        const auto utf8 = options.cache_directory.u8string();
        result.emplace(ov::cache_dir.name(), std::string(utf8.begin(), utf8.end()));
    }
    return result;
}

} // namespace

struct CompiledModel::Impl {
    Impl(const std::filesystem::path& path, const CompileOptions& options) {
        auto graph = core.read_model(path);
        if (!options.input_shapes.empty()) {
            std::map<std::size_t, ov::PartialShape> shapes;
            for (std::size_t i = 0; i < options.input_shapes.size(); ++i) {
                shapes.emplace(i, ov::PartialShape(runtime_shape(options.input_shapes[i])));
            }
            graph->reshape(shapes);
        }
        // 显式选择 NPU，由插件完成图优化、精度安排和设备资源分配。
        compiled = core.compile_model(graph, "NPU", properties(options));
        input_ports = describe(compiled.inputs());
        output_ports = describe(compiled.outputs());
    }

    // 逆序销毁先释放编译模型，再释放持有设备插件的 Core。
    ov::Core core;
    ov::CompiledModel compiled;
    std::vector<Port> input_ports, output_ports;
};

struct Request::Impl {
    explicit Impl(std::shared_ptr<CompiledModel::Impl> model)
        : owner(std::move(model)), request(owner->compiled.create_infer_request()) {}

    // 请求使用运行时自身的输入输出缓冲，其生命周期先于共享模型结束。
    std::shared_ptr<CompiledModel::Impl> owner;
    ov::InferRequest request;
};

CompiledModel::CompiledModel(const std::filesystem::path& path, CompileOptions options)
    : impl_(std::make_shared<Impl>(path, options)) {}

const std::vector<Port>& CompiledModel::inputs() const { return impl_->input_ports; }
const std::vector<Port>& CompiledModel::outputs() const { return impl_->output_ports; }

std::size_t CompiledModel::optimal_request_count() const {
    return impl_->compiled.get_property(ov::optimal_number_of_infer_requests);
}

Request CompiledModel::create_request() const { return Request(*this); }

Request::Request(const CompiledModel& model) : impl_(std::make_unique<Impl>(model.impl_)) {}
Request::~Request() = default;
Request::Request(Request&&) noexcept = default;
Request& Request::operator=(Request&&) noexcept = default;

void Request::set_input(std::size_t index, const Tensor& input) {
    auto dense = input.detach().to(Device::cpu()).contiguous();
    const ov::Tensor source(element_type(dense.dtype()), runtime_shape(dense.shape()), dense.data<void>());
    // 只在本次调用中借用 Ctorch 地址，copy_to 完成后由请求独立持有输入值。
    source.copy_to(impl_->request.get_input_tensor(index));
}

void Request::infer() { impl_->request.infer(); }
void Request::start_async() { impl_->request.start_async(); }
void Request::wait() { impl_->request.wait(); }
bool Request::ready() { return impl_->request.wait_for(std::chrono::milliseconds{0}); }

Tensor Request::output(std::size_t index) const {
    return copy_output(impl_->request.get_output_tensor(index));
}

std::vector<Tensor> Request::outputs() const {
    std::vector<Tensor> result;
    result.reserve(impl_->owner->output_ports.size());
    for (std::size_t i = 0; i < impl_->owner->output_ports.size(); ++i) result.push_back(output(i));
    return result;
}

void Request::output_into(std::size_t index, Tensor destination) const {
    // 普通 data 接口提供只读地址；此显式写入入口借用调用方拥有的可写存储。
    // OpenVINO 按源布局复制到连续目标，返回时结束借用，后续执行由请求独立管理。
    ov::Tensor target(element_type(destination.dtype()), runtime_shape(destination.shape()),
                      const_cast<void*>(destination.data<void>()));
    impl_->request.get_output_tensor(index).copy_to(target);
}

void Request::reset_state() {
    for (auto state : impl_->request.query_state()) state.reset();
}

} // namespace dl::npu

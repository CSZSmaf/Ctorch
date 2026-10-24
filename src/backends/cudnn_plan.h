#pragma once

#include "backends/backend.h"
#include "core/layout.h"

#include <cudnn_frontend.h>

#include <array>
#include <compare>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dl::detail::cuda::dnn {

namespace fe = cudnn_frontend;

inline void check(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuDNN: ") + cudnnGetErrorString(status));
    }
}

inline void check(fe::error_t status) {
    if (status.is_bad()) {
        throw std::runtime_error("cuDNN graph: " + status.get_message());
    }
}

/// 计划由方向、形状、卷积几何和输入精度确定；物理布局由精度策略确定。
struct PlanKey {
    ConvolutionPass pass;
    Shape first, second, output;
    // 两个空间轴的 stride、padding、dilation 后接分组数。
    std::array<Index, 7> geometry;
    DType dtype = DType::f32;
    auto operator<=>(const PlanKey&) const = default;
};

inline Shape tensor_strides(const Shape& shape, DType dtype) {
    // 低精度图采用 NHWC / KRSC 物理存储，逻辑维度继续保持 NCHW / KCRS。
    if (is_low_precision(dtype)) return {shape[1] * shape[2] * shape[3], 1, shape[3] * shape[1], shape[1]};
    return contiguous_strides(shape);
}

inline fe::DataType_t storage_type(DType dtype) {
    if (dtype == DType::f16) return fe::DataType_t::HALF;
    if (dtype == DType::bf16) return fe::DataType_t::BFLOAT16;
    return fe::DataType_t::FLOAT;
}

template <class Attributes> Attributes convolution_attributes(const PlanKey& key) {
    Attributes attributes;
    attributes.set_stride({key.geometry[0], key.geometry[1]})
        .set_padding({key.geometry[2], key.geometry[3]})
        .set_dilation({key.geometry[4], key.geometry[5]})
        .set_convolution_mode(fe::ConvolutionMode_t::CROSS_CORRELATION);
    return attributes;
}

/// 图拥有描述与执行计划；调用方拥有 handle、操作数和按字节计的临时工作区。
struct Plan {
    fe::graph::Graph graph;
    int64_t workspace_bytes = 0;

    Plan(cudnnHandle_t handle, const PlanKey& key) {
        graph.set_io_data_type(fe::DataType_t::FLOAT)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto tensor = [&](const Shape& shape, int64_t uid) {
            return graph.tensor(fe::graph::Tensor_attributes()
                .set_dim(shape).set_stride(tensor_strides(shape, key.dtype))
                .set_data_type(storage_type(key.dtype)).set_uid(uid));
        };
        auto first = tensor(key.first, 1);
        auto second = tensor(key.second, 2);
        std::shared_ptr<fe::graph::Tensor_attributes> output;
        switch (key.pass) {
        case ConvolutionPass::forward:
            output = graph.conv_fprop(first, second,
                convolution_attributes<fe::graph::Conv_fprop_attributes>(key));
            break;
        case ConvolutionPass::input_gradient:
            output = graph.conv_dgrad(first, second,
                convolution_attributes<fe::graph::Conv_dgrad_attributes>(key));
            break;
        case ConvolutionPass::weight_gradient:
            output = graph.conv_wgrad(first, second,
                convolution_attributes<fe::graph::Conv_wgrad_attributes>(key));
            break;
        }
        // dgrad 的输出尺寸保留原输入形状，因此 stride 大于一时也恢复同一张量空间。
        // 分组由图中完整通道数与滤波器每组通道数的比例表达。
        output->set_dim(key.output).set_stride(tensor_strides(key.output, key.dtype))
            .set_data_type(storage_type(key.dtype)).set_output(true).set_uid(3);
        check(graph.validate());
        check(graph.build_operation_graph(handle));
        check(graph.create_execution_plans({fe::HeurMode_t::A}));
        std::vector notes{fe::NumericalNote_t::DOWN_CONVERT_INPUTS, fe::NumericalNote_t::REDUCED_PRECISION_REDUCTION};
        if (!is_low_precision(key.dtype)) notes.push_back(fe::NumericalNote_t::TENSOR_CORE);
        graph.deselect_numeric_notes(notes);
        check(graph.check_support(handle));
        check(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
        check(graph.get_workspace_size(workspace_bytes));
    }
};

} // namespace dl::detail::cuda::dnn

#pragma once

#include "backends/cudnn_plan.h"

#include <cmath>

namespace dl::detail::cuda::dnn {

enum class AttentionPass { inference, forward, backward };

/// 输入统一为连续 BHSD，缓存只依赖计算规格；随机 key 和张量地址在每次执行时绑定。
struct AttentionPlanKey {
    AttentionPass pass;
    std::array<Shape, 3> inputs;
    Shape bias;
    DType dtype;
    float dropout;
    bool causal;
    auto operator<=>(const AttentionPlanKey&) const = default;
};

struct AttentionPlan {
    enum Id : int64_t { Q = 1, K, V, O, Stats, DO, DQ, DK, DV, Bias, Seed, Offset };
    fe::graph::Graph graph;
    int64_t workspace_bytes = 0;

    AttentionPlan(cudnnHandle_t handle, const AttentionPlanKey& key) {
        graph.set_io_data_type(storage_type(key.dtype))
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto tensor = [&](const Shape& shape, Id id) {
            return graph.tensor(fe::graph::Tensor_attributes()
                .set_dim(shape).set_stride(contiguous_strides(shape)).set_uid(id));
        };
        auto q = tensor(key.inputs[0], Q), k = tensor(key.inputs[1], K), v = tensor(key.inputs[2], V);
        auto output_shape = key.inputs[0];
        output_shape.back() = key.inputs[2].back();
        auto stats_shape = key.inputs[0];
        stats_shape.back() = 1;

        auto configure = [&](auto attributes) {
            attributes.set_attn_scale(1.0f / std::sqrt(static_cast<float>(key.inputs[0].back())));
            if (key.causal)
                attributes.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                    .set_diagonal_band_right_bound(0);
            if (!key.bias.empty()) attributes.set_bias(tensor(key.bias, Bias));
            if (key.dropout > 0.0f) {
                auto seed = tensor({1, 1, 1, 1}, Seed);
                auto offset = tensor({1, 1, 1, 1}, Offset);
                seed->set_data_type(fe::DataType_t::INT64).set_is_pass_by_value(true);
                offset->set_data_type(fe::DataType_t::INT64).set_is_pass_by_value(true);
                attributes.set_dropout(key.dropout, seed, offset);
            }
            return attributes;
        };
        auto output = [](auto value, const Shape& shape, Id id) {
            value->set_dim(shape).set_stride(contiguous_strides(shape)).set_output(true).set_uid(id);
        };

        if (key.pass == AttentionPass::backward) {
            auto o = tensor(output_shape, O), gradient = tensor(output_shape, DO);
            auto stats = tensor(stats_shape, Stats);
            stats->set_data_type(fe::DataType_t::FLOAT);
            auto gradients = graph.sdpa_backward(q, k, v, o, gradient, stats,
                configure(fe::graph::SDPA_backward_attributes()));
            output(gradients[0], key.inputs[0], DQ);
            output(gradients[1], key.inputs[1], DK);
            output(gradients[2], key.inputs[2], DV);
        } else {
            auto attributes = configure(fe::graph::SDPA_attributes());
            attributes.set_is_inference(key.pass == AttentionPass::inference);
            auto [o, stats] = graph.sdpa(q, k, v, attributes);
            output(o, output_shape, O);
            if (key.pass == AttentionPass::forward) {
                output(stats, stats_shape, Stats);
                stats->set_data_type(fe::DataType_t::FLOAT);
            }
        }
        check(graph.validate());
        check(graph.build_operation_graph(handle));
        check(graph.create_execution_plans({fe::HeurMode_t::A}));
        check(graph.check_support(handle));
        check(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
        check(graph.get_workspace_size(workspace_bytes));
    }
};

} // namespace dl::detail::cuda::dnn

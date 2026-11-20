#include "dl/onnx_c_api.h"
#include "dl/onnx.h"
#include "api/common.h"

extern "C" dl_status dl_onnx_save(const char* path, const dl_graph* graph,
    const dl_plan_value* outputs, size_t count, const dl_onnx_binding* weights, size_t weight_count) {
    return dl::capi::invoke([&] {
        std::vector<dl::plan::Value> values;
        for (std::size_t i = 0; i < count; ++i) values.push_back({outputs[i].index});
        std::vector<dl::onnx::Binding> constants;
        for (std::size_t i = 0; i < weight_count; ++i) constants.push_back({{weights[i].input.index}, weights[i].value->value});
        dl::onnx::save(dl::capi::utf8_path(path), graph->value, values, constants);
    });
}

extern "C" dl_status dl_onnx_save_decoder(const char* path, const dl_module* decoder,
    const dl_module* embedding, const dl_module* output) {
    return dl::capi::invoke([&] {
        const auto& model = dynamic_cast<const dl::nn::TransformerDecoder&>(*decoder->value);
        auto file = dl::capi::utf8_path(path);
        if (!embedding && !output) return dl::onnx::save_decoder(file, model);
        if (!embedding || !output) throw std::invalid_argument("decoder export needs both embedding and output layers");
        dl::onnx::save_decoder(file, dynamic_cast<const dl::nn::Embedding&>(*embedding->value),
            model, dynamic_cast<const dl::nn::Linear&>(*output->value));
    });
}

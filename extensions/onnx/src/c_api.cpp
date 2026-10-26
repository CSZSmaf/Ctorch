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
        dl::onnx::save(std::filesystem::path(std::u8string_view(reinterpret_cast<const char8_t*>(path))),
            graph->value, values, constants);
    });
}

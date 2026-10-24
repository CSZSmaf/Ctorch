#include "dl/npu_c_api.h"
#include "dl/npu.h"
#include "api/common.h"

struct dl_npu_model { dl::npu::CompiledModel value; };
struct dl_npu_request { dl::npu::Request value; };

namespace {

dl::npu::CompileOptions compile_options(const dl_npu_compile_options* source) {
    dl::npu::CompileOptions result;
    if (!source) return result;
    result.performance = static_cast<dl::npu::Performance>(source->performance);
    if (source->cache_directory) result.cache_directory = dl::capi::utf8_path(source->cache_directory);
    result.input_shapes.reserve(source->input_shape_count);
    for (std::size_t i = 0; i < source->input_shape_count; ++i) {
        result.input_shapes.push_back(dl::capi::convert(source->input_shapes[i]));
    }
    return result;
}

dl_npu_port port_view(const dl::npu::Port& port) {
    return {port.name.c_str(), static_cast<dl_dtype>(port.dtype), {port.shape.data(), port.shape.size()}};
}

} // namespace

extern "C" {

dl_npu_compile_options dl_npu_default_compile_options(void) {
    const dl::npu::CompileOptions options;
    return {static_cast<dl_npu_performance>(options.performance), nullptr, nullptr, 0};
}

dl_status dl_npu_model_create(const char* path, const dl_npu_compile_options* options, dl_npu_model** output) {
    return dl::capi::make_handle(output, [&] {
        return dl::npu::CompiledModel(dl::capi::utf8_path(path), compile_options(options));
    });
}

dl_status dl_npu_model_retain(const dl_npu_model* model, dl_npu_model** output) {
    return dl::capi::make_handle(output, [&] { return model->value; });
}

void dl_npu_model_release(dl_npu_model* model) { delete model; }
std::size_t dl_npu_model_input_count(const dl_npu_model* model) { return model->value.inputs().size(); }
std::size_t dl_npu_model_output_count(const dl_npu_model* model) { return model->value.outputs().size(); }

dl_npu_port dl_npu_model_input_port(const dl_npu_model* model, std::size_t index) {
    return port_view(model->value.inputs()[index]);
}

dl_npu_port dl_npu_model_output_port(const dl_npu_model* model, std::size_t index) {
    return port_view(model->value.outputs()[index]);
}

dl_status dl_npu_model_optimal_request_count(const dl_npu_model* model, std::size_t* count) {
    return dl::capi::invoke([&] { *count = model->value.optimal_request_count(); });
}

dl_status dl_npu_request_create(const dl_npu_model* model, dl_npu_request** output) {
    return dl::capi::make_handle(output, [&] { return model->value.create_request(); });
}

void dl_npu_request_release(dl_npu_request* request) { delete request; }

dl_status dl_npu_request_set_input(dl_npu_request* request, std::size_t index, const dl_tensor* input) {
    return dl::capi::invoke([&] { request->value.set_input(index, input->value); });
}

dl_status dl_npu_request_infer(dl_npu_request* request) {
    return dl::capi::invoke([&] { request->value.infer(); });
}

dl_status dl_npu_request_start_async(dl_npu_request* request) {
    return dl::capi::invoke([&] { request->value.start_async(); });
}

dl_status dl_npu_request_wait(dl_npu_request* request) {
    return dl::capi::invoke([&] { request->value.wait(); });
}

dl_status dl_npu_request_ready(dl_npu_request* request, std::int32_t* ready) {
    return dl::capi::invoke([&] { *ready = request->value.ready(); });
}

dl_status dl_npu_request_output(const dl_npu_request* request, std::size_t index, dl_tensor** output) {
    return dl::capi::make_handle(output, [&] { return request->value.output(index); });
}

dl_status dl_npu_request_output_into(const dl_npu_request* request, std::size_t index, dl_tensor* destination) {
    return dl::capi::invoke([&] { request->value.output_into(index, destination->value); });
}

dl_status dl_npu_request_reset_state(dl_npu_request* request) {
    return dl::capi::invoke([&] { request->value.reset_state(); });
}

} // extern "C"

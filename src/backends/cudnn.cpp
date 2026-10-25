#include "backends/cudnn.h"
#include "backends/cudnn_plan.h"
#include "backends/cudnn_attention_plan.h"
#include "dl/random.h"

#include <bit>
#include <map>
#include <mutex>
#include <unordered_map>

namespace dl::detail::cuda {
namespace {

struct DnnDeleter {
    int device;
    void operator()(cudnnHandle_t handle) const noexcept {
        int previous = 0;
        cudaGetDevice(&previous);
        if (previous != device) cudaSetDevice(device);
        cudnnDestroy(handle);
        if (previous != device) cudaSetDevice(previous);
    }
};

cudnnHandle_t create_handle() {
    cudnnHandle_t handle;
    dnn::check(cudnnCreate(&handle));
    return handle;
}

} // namespace

struct DnnContext {
    DnnContext(Device device, cudaStream_t stream, cudaMemPool_t pool)
        : handle(create_handle(), DnnDeleter{device.index}), stream(stream), pool(pool),
          workspace(nullptr, AsyncBufferDeleter{device.index, stream}) {
        dnn::check(cudnnSetStream(handle.get(), stream));
    }

    // 宿主 mutex 保护容量与提交顺序，同一 stream 让相邻执行依次使用同一个地址。
    void* workspace_for(std::size_t bytes) {
        if (bytes > capacity) {
            workspace.reset();
            capacity = 0;
            void* address;
            check(cudaMallocFromPoolAsync(&address, bytes, pool, stream));
            workspace.reset(address);
            capacity = bytes;
        }
        return bytes == 0 ? nullptr : workspace.get();
    }

    std::unique_ptr<std::remove_pointer_t<cudnnHandle_t>, DnnDeleter> handle;
    std::mutex mutex;
    std::map<dnn::PlanKey, std::unique_ptr<dnn::Plan>> plans;
    std::map<dnn::AttentionPlanKey, std::unique_ptr<dnn::AttentionPlan>> attention_plans;
    cudaStream_t stream;
    cudaMemPool_t pool;
    // 仅借用 Context 的 stream/pool，工作区不通过 Storage 反向持有 Context。
    std::unique_ptr<void, AsyncBufferDeleter> workspace;
    std::size_t capacity = 0;
};

std::shared_ptr<DnnContext> make_dnn_context(Device device, cudaStream_t stream, cudaMemPool_t pool) {
    return std::make_shared<DnnContext>(device, stream, pool);
}

void release_dnn_workspace(DnnContext& context) {
    const std::lock_guard lock(context.mutex);
    context.workspace.reset();
    context.capacity = 0;
}

TensorData prepare_convolution(const TensorData& input, DType dtype) {
    const bool channels_last = is_low_precision(dtype);
    auto view = channels_last ? numeric::permute(input, {0, 2, 3, 1}) : input;
    if (view.dtype != dtype) {
        // cast 从源布局直接写入连续目标，转换和重排由同一次数据遍历完成。
        view = numeric::cast(view, dtype);
    } else if (!view.layout.is_contiguous() || view.layout.offset != 0) {
        view = numeric::clone(view);
    }
    return channels_last ? numeric::permute(view, {0, 3, 1, 2}) : view;
}

TensorData convolution(ConvolutionPass pass, const TensorData& first, const TensorData& second,
                       Shape output_shape, const Conv2dOptions& options) {
    const auto output_dtype = accumulation_dtype(first.dtype);
    // 空 batch 的前向与输入梯度为空，权重梯度是空集合的和。
    if (numel(output_shape) == 0 || first.layout.numel() == 0 || second.layout.numel() == 0) {
        return numeric::full(std::move(output_shape), 0.0, output_dtype, first.device());
    }
    Execution execution(first);
    auto result = empty_data(std::move(output_shape), first.dtype, first.device());
    result.layout.strides = dnn::tensor_strides(result.layout.sizes, first.dtype);
    dnn::PlanKey key{pass, first.layout.sizes, second.layout.sizes, result.layout.sizes,
        {options.stride.height, options.stride.width, options.padding.height, options.padding.width,
         options.dilation.height, options.dilation.width, options.groups}, first.dtype};
    auto& context = *execution.context.dnn;
    const std::lock_guard lock(context.mutex);
    auto position = context.plans.find(key);
    if (position == context.plans.end()) {
        auto plan = std::make_unique<dnn::Plan>(context.handle.get(), key);
        position = context.plans.emplace(std::move(key), std::move(plan)).first;
    }
    auto& plan = *position->second;
    auto* workspace = context.workspace_for(static_cast<std::size_t>(plan.workspace_bytes));
    std::unordered_map<int64_t, void*> pointers{
        {1, const_cast<void*>(first.address())}, {2, const_cast<void*>(second.address())}, {3, result.mutable_address()}};
    dnn::check(plan.graph.execute(context.handle.get(), pointers, workspace));
    // 库卷积以输入精度写回，FP32 计算的结果在库边界舍入，再展开到框架的累计精度。
    return numeric::cast(result, output_dtype);
}

namespace {

void execute_attention(const std::array<TensorData, 3>& inputs, const std::optional<TensorData>& bias,
    AttentionComputeOptions options, RandomKey random, dnn::AttentionPass pass,
    std::unordered_map<int64_t, void*> pointers) {
    using Id = dnn::AttentionPlan;
    for (std::size_t i = 0; i < inputs.size(); ++i)
        pointers.emplace(Id::Q + static_cast<int64_t>(i), const_cast<void*>(inputs[i].address()));
    if (bias) pointers.emplace(Id::Bias, const_cast<void*>(bias->address()));
    // pass-by-value 标量由本次执行读取；反向绑定同一组位模式，不再消费 Generator。
    auto seed = std::bit_cast<int64_t>(random.seed), offset = std::bit_cast<int64_t>(random.stream);
    const auto probability = static_cast<float>(options.dropout);
    if (probability > 0.0f) {
        pointers.emplace(Id::Seed, &seed);
        pointers.emplace(Id::Offset, &offset);
    }
    dnn::AttentionPlanKey key{pass,
        {inputs[0].layout.sizes, inputs[1].layout.sizes, inputs[2].layout.sizes},
        bias ? bias->layout.sizes : Shape{}, inputs[0].dtype, probability, options.causal};
    Execution execution(inputs[0]);
    auto& context = *execution.context.dnn;
    const std::lock_guard lock(context.mutex);
    auto position = context.attention_plans.find(key);
    if (position == context.attention_plans.end()) {
        auto plan = std::make_unique<dnn::AttentionPlan>(context.handle.get(), key);
        position = context.attention_plans.emplace(std::move(key), std::move(plan)).first;
    }
    auto& plan = *position->second;
    auto* workspace = context.workspace_for(static_cast<std::size_t>(plan.workspace_bytes));
    dnn::check(plan.graph.execute(context.handle.get(), pointers, workspace));
}

} // namespace

AttentionResult attention_forward(const std::array<TensorData, 3>& inputs,
    const std::optional<TensorData>& bias, AttentionComputeOptions options, RandomKey random, bool save_statistics) {
    using Id = dnn::AttentionPlan;
    auto shape = inputs[0].layout.sizes;
    shape.back() = inputs[2].layout.sizes.back();
    AttentionResult result{empty_data(shape, inputs[0].dtype, inputs[0].device()), std::nullopt};
    std::unordered_map<int64_t, void*> pointers{{Id::O, result.output.mutable_address()}};
    if (save_statistics) {
        shape.back() = 1;
        result.statistics = empty_data(std::move(shape), DType::f32, inputs[0].device());
        pointers.emplace(Id::Stats, result.statistics->mutable_address());
    }
    execute_attention(inputs, bias, options, random,
        save_statistics ? dnn::AttentionPass::forward : dnn::AttentionPass::inference, std::move(pointers));
    return result;
}

std::array<TensorData, 3> attention_backward(const std::array<TensorData, 3>& inputs,
    const TensorData& output, const TensorData& statistics, const TensorData& grad,
    const std::optional<TensorData>& bias, AttentionComputeOptions options, RandomKey random) {
    using Id = dnn::AttentionPlan;
    std::array<TensorData, 3> result{
        empty_data(inputs[0].layout.sizes, inputs[0].dtype, inputs[0].device()),
        empty_data(inputs[1].layout.sizes, inputs[1].dtype, inputs[1].device()),
        empty_data(inputs[2].layout.sizes, inputs[2].dtype, inputs[2].device())};
    std::unordered_map<int64_t, void*> pointers{
        {Id::O, const_cast<void*>(output.address())}, {Id::Stats, const_cast<void*>(statistics.address())},
        {Id::DO, const_cast<void*>(grad.address())}, {Id::DQ, result[0].mutable_address()},
        {Id::DK, result[1].mutable_address()}, {Id::DV, result[2].mutable_address()}};
    execute_attention(inputs, bias, options, random, dnn::AttentionPass::backward, std::move(pointers));
    return result;
}

} // namespace dl::detail::cuda

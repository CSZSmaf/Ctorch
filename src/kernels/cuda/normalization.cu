#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/storage.cuh"

namespace dl::detail::cuda {
namespace {

/// 一行在一个 block 内完成统计与仿射；半精度值在寄存器中展开为 FP32。
template <class T>
__global__ void normalization_kernel(const T* input, const float* weight, const float* bias,
    float* output, float* statistics, Index rows, Index width, float epsilon, bool centered,
    IndexMap input_layout, IndexMap weight_layout, IndexMap bias_layout) {
    __shared__ float shared[block_size];
    const float inverse_width = 1.0f / static_cast<float>(width);
    for (Index row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto base = row * width;
        float mean = 0.0f;
        if (centered) {
            for (Index col = threadIdx.x; col < width; col += blockDim.x)
                mean += storage_cast<float>(input[input_layout.offset(base + col)]);
            mean = block_reduce<false>(mean, shared) * inverse_width;
        }
        float squares = 0.0f;
        for (Index col = threadIdx.x; col < width; col += blockDim.x) {
            const float value = storage_cast<float>(input[input_layout.offset(base + col)]) - mean;
            squares += value * value;
        }
        const float inverse_std = rsqrtf(block_reduce<false>(squares, shared) * inverse_width + epsilon);
        if (statistics && threadIdx.x == 0) {
            statistics[row] = mean;
            statistics[rows + row] = inverse_std;
        }
        for (Index col = threadIdx.x; col < width; col += blockDim.x) {
            float value = (storage_cast<float>(input[input_layout.offset(base + col)]) - mean) * inverse_std;
            if (weight) value *= weight[weight_layout.offset(col)];
            if (bias) value += bias[bias_layout.offset(col)];
            output[base + col] = value;
        }
    }
}

template <class T>
__global__ void normalization_input_backward_kernel(const float* grad, const T* input,
    const float* statistics, const float* weight, float* output, Index rows, Index width,
    bool centered, bool accumulate, IndexMap grad_layout, IndexMap input_layout, IndexMap weight_layout) {
    __shared__ float shared[block_size];
    const float inverse_width = 1.0f / static_cast<float>(width);
    for (Index row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto base = row * width;
        const float mean = statistics[row], inverse_std = statistics[rows + row];
        float average = 0.0f, projection = 0.0f;
        for (Index col = threadIdx.x; col < width; col += blockDim.x) {
            const float z = (storage_cast<float>(input[input_layout.offset(base + col)]) - mean) * inverse_std;
            float value = grad[grad_layout.offset(base + col)];
            if (weight) value *= weight[weight_layout.offset(col)];
            if (centered) average += value;
            projection += value * z;
        }
        if (centered) average = block_reduce<false>(average, shared) * inverse_width;
        projection = block_reduce<false>(projection, shared) * inverse_width;
        for (Index col = threadIdx.x; col < width; col += blockDim.x) {
            const float z = (storage_cast<float>(input[input_layout.offset(base + col)]) - mean) * inverse_std;
            float value = grad[grad_layout.offset(base + col)];
            if (weight) value *= weight[weight_layout.offset(col)];
            const float contribution = (value - average - z * projection) * inverse_std;
            output[base + col] = (accumulate ? output[base + col] : 0.0f) + contribution;
        }
    }
}

/// 相邻线程负责相邻列，跨行读取保持合并访问；每个分区独占一份仿射梯度。
template <class T>
__global__ void normalization_parameter_kernel(const float* grad, const T* input,
    const float* statistics, float* dw, float* db, Index rows, Index width, Index parts,
    bool add_weight, bool add_bias, IndexMap grad_layout, IndexMap input_layout) {
    const Index columns = (width + blockDim.x - 1) / blockDim.x;
    for (Index task = blockIdx.x; task < parts * columns; task += gridDim.x) {
        const auto part = task / columns;
        const auto col = (task % columns) * blockDim.x + threadIdx.x;
        if (col >= width) continue;
        float weight_sum = 0.0f, bias_sum = 0.0f;
        for (Index row = part; row < rows; row += parts) {
            const auto index = row * width + col;
            const float value = grad[grad_layout.offset(index)];
            if (dw) {
                const float z = (storage_cast<float>(input[input_layout.offset(index)]) - statistics[row])
                    * statistics[rows + row];
                weight_sum += value * z;
            }
            if (db) bias_sum += value;
        }
        if (dw) dw[part * width + col] = (add_weight ? dw[part * width + col] : 0.0f) + weight_sum;
        if (db) db[part * width + col] = (add_bias ? db[part * width + col] : 0.0f) + bias_sum;
    }
}

/// 按固定分区顺序合并，避免多个 block 原子写入同一参数。
__global__ void normalization_parameter_sum_kernel(const float* weight_parts, const float* bias_parts,
    float* dw, float* db, Index width, Index parts, bool add_weight, bool add_bias) {
    for (Index col = blockIdx.x * blockDim.x + threadIdx.x; col < width; col += gridDim.x * blockDim.x) {
        float weight_sum = 0.0f, bias_sum = 0.0f;
        for (Index part = 0; part < parts; ++part) {
            if (dw) weight_sum += weight_parts[part * width + col];
            if (db) bias_sum += bias_parts[part * width + col];
        }
        if (dw) dw[col] = (add_weight ? dw[col] : 0.0f) + weight_sum;
        if (db) db[col] = (add_bias ? db[col] : 0.0f) + bias_sum;
    }
}

} // namespace

void normalization(const TensorData& input, const std::optional<TensorData>& weight,
    const std::optional<TensorData>& bias, Index width, double epsilon,
    numeric::NormalizationKind kind, numeric::NormalizationResult& result) {
    const auto rows = input.layout.numel() / width;
    if (rows == 0) return;
    Execution execution(input);
    visit_storage(input.dtype, [&]<class T>() {
        normalization_kernel<<<row_grid_size(rows), block_size, 0, execution.stream()>>>(
            input.data<T>(), weight ? weight->data<float>() : nullptr, bias ? bias->data<float>() : nullptr,
            result.output.mutable_data<float>(), result.statistics ? result.statistics->mutable_data<float>() : nullptr,
            rows, width, static_cast<float>(epsilon), kind == numeric::NormalizationKind::layer,
            IndexMap(input.layout), weight ? IndexMap(weight->layout) : IndexMap{},
            bias ? IndexMap(bias->layout) : IndexMap{});
    });
    check(cudaGetLastError());
}

void normalization_backward(const TensorData& grad, const TensorData& input,
    const TensorData& statistics, const std::optional<TensorData>& weight, Index width,
    numeric::NormalizationKind kind, numeric::NormalizationGradients& result) {
    const auto rows = input.layout.numel() / width;
    Execution execution(input);
    if (result.input && rows > 0) {
        visit_storage(input.dtype, [&]<class T>() {
            normalization_input_backward_kernel<<<row_grid_size(rows), block_size, 0, execution.stream()>>>(
                grad.data<float>(), input.data<T>(), statistics.data<float>(),
                weight ? weight->data<float>() : nullptr, result.input->mutable_data<float>(), rows, width,
                kind == numeric::NormalizationKind::layer, result.accumulate[0], IndexMap(grad.layout), IndexMap(input.layout),
                weight ? IndexMap(weight->layout) : IndexMap{});
        });
        check(cudaGetLastError());
    }
    if (result.weight || result.bias) {
        // 小批量直接写最终梯度；大批量最多保留 128 个分区，工作区不随行数无限增长。
        const auto parts = std::max(Index{1}, std::min((rows + 255) / 256, Index{128}));
        auto* dw = result.weight ? result.weight->mutable_data<float>() : nullptr;
        auto* db = result.bias ? result.bias->mutable_data<float>() : nullptr;
        std::optional<TensorData> weight_parts, bias_parts;
        if (parts > 1 && dw) weight_parts = empty_data({parts, width}, DType::f32, input.device());
        if (parts > 1 && db) bias_parts = empty_data({parts, width}, DType::f32, input.device());
        auto* partial_dw = weight_parts ? weight_parts->mutable_data<float>() : dw;
        auto* partial_db = bias_parts ? bias_parts->mutable_data<float>() : db;
        const auto tasks = parts * ((width + block_size - 1) / block_size);
        visit_storage(input.dtype, [&]<class T>() {
            normalization_parameter_kernel<<<row_grid_size(tasks), block_size, 0, execution.stream()>>>(
                grad.data<float>(), input.data<T>(), statistics.data<float>(), partial_dw, partial_db,
                rows, width, parts, parts == 1 && result.accumulate[1], parts == 1 && result.accumulate[2],
                IndexMap(grad.layout), IndexMap(input.layout));
        });
        check(cudaGetLastError());
        if (parts > 1) {
            normalization_parameter_sum_kernel<<<grid_size(width), block_size, 0, execution.stream()>>>(
                partial_dw, partial_db, dw, db, width, parts, result.accumulate[1], result.accumulate[2]);
            check(cudaGetLastError());
        }
    }
}

} // namespace dl::detail::cuda

#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "kernels/cuda/vector.cuh"

namespace dl::detail::cuda {
namespace {

constexpr int optimizer_batch_capacity = 32;
constexpr int optimizer_values_per_thread = 4;

struct SGDSegment {
    using Arithmetic = SGDArithmetic;
    float* parameter;
    const float* gradient;
    float* velocity;
    Index count;
    int begin, end;
    numeric::SGDStep step;

    __device__ void apply(const Arithmetic& update, Index index) const {
        update(parameter[index], gradient[index], velocity ? velocity + index : nullptr);
    }
};

struct AdamSegment {
    using Arithmetic = AdamArithmetic;
    float* parameter;
    const float* gradient;
    float* first;
    float* second;
    Index count;
    int begin, end;
    numeric::AdamStep step;

    __device__ void apply(const Arithmetic& update, Index index) const {
        update(parameter[index], gradient[index], first[index], second[index]);
    }
};

struct RMSpropSegment {
    using Arithmetic = RMSpropArithmetic;
    float* parameter;
    const float* gradient;
    float* square_average;
    float* momentum;
    float* gradient_average;
    Index count;
    int begin, end;
    numeric::RMSpropStep step;

    __device__ void apply(const Arithmetic& update, Index index) const {
        update(parameter[index], gradient[index], square_average[index],
            momentum ? momentum + index : nullptr, gradient_average ? gradient_average + index : nullptr);
    }
};

struct AdagradSegment {
    using Arithmetic = AdagradArithmetic;
    float* parameter;
    const float* gradient;
    float* sum;
    Index count;
    int begin, end;
    numeric::AdagradStep step;

    __device__ void apply(const Arithmetic& update, Index index) const {
        update(parameter[index], gradient[index], sum[index]);
    }
};

template <class Segment>
struct OptimizerBatch {
    Segment segments[optimizer_batch_capacity]{};
    int count = 0;
    int blocks = 0;
};

template <class Segment>
__global__ void optimizer_batch_kernel(const OptimizerBatch<Segment> batch) {
    const auto block = static_cast<int>(blockIdx.x);
    const auto segment = batch.segments[segment_index(batch, block)];
    const typename Segment::Arithmetic update(segment.step);
    const auto stride = static_cast<Index>(segment.end - segment.begin) * blockDim.x * optimizer_values_per_thread;
    for (Index base = static_cast<Index>(block - segment.begin) * blockDim.x * optimizer_values_per_thread + threadIdx.x;
         base < segment.count; base += stride) {
        // 每线程更新四个相隔 blockDim.x 的元素，各轮 warp 读写仍连续，描述和系数可以复用。
        #pragma unroll
        for (int item = 0; item < optimizer_values_per_thread; ++item) {
            const auto index = base + item * blockDim.x;
            if (index < segment.count) segment.apply(update, index);
        }
    }
}

/// 连续段与一般布局按输入顺序提交；算法只提供布局选择、描述符和单参数调用。
template <class Segment, class Update, class Dense, class Describe, class Strided>
void submit_optimizer_updates(std::span<const Update> updates, Dense dense, Describe describe, Strided strided) {
    Execution execution(*updates.front().parameter);
    OptimizerBatch<Segment> batch;
    auto launch = [&] {
        optimizer_batch_kernel<Segment><<<batch.blocks, block_size, 0, execution.stream()>>>(batch);
        check(cudaGetLastError());
        batch = {};
    };
    for (const auto& update : updates) {
        const auto count = update.parameter->layout.numel();
        if (count == 0) continue;
        if (!dense(update)) {
            if (batch.count != 0) launch();
            strided(update, execution.stream());
            check(cudaGetLastError());
            continue;
        }
        // 表随 kernel 参数传入，设备侧直接读取各张量和选项，无需单独的元信息缓冲。
        const int end = batch.blocks + grid_size((count + optimizer_values_per_thread - 1) / optimizer_values_per_thread);
        batch.segments[batch.count++] = describe(update, count, batch.blocks, end);
        batch.blocks = end;
        if (batch.count == optimizer_batch_capacity) launch();
    }
    if (batch.count != 0) launch();
}

} // namespace

void axpy(TensorData& target, const TensorData& input, double alpha) {
    const auto count = input.layout.numel();
    if (count != 0) {
        Execution execution(input);
        axpy_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
            target.mutable_data<float>(), input.data<float>(), count,
            IndexMap(target.layout), IndexMap(input.layout), static_cast<float>(alpha));
        check(cudaGetLastError());
    }
}

void momentum(TensorData& buffer, const TensorData& gradient, double decay, double scale) {
    const auto count = buffer.layout.numel();
    if (count != 0) {
        Execution execution(buffer);
        momentum_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
            buffer.mutable_data<float>(), gradient.data<float>(), count,
            IndexMap(buffer.layout), IndexMap(gradient.layout), static_cast<float>(decay), static_cast<float>(scale));
        check(cudaGetLastError());
    }
}

void sgd(std::span<const numeric::SGDUpdate> updates) {
    submit_optimizer_updates<SGDSegment>(updates,
        [](const auto& update) {
            return update.parameter->layout.is_contiguous() && update.gradient->layout.is_contiguous()
                && (!update.velocity || update.velocity->layout.is_contiguous());
        },
        [](const numeric::SGDUpdate& update, Index count, int begin, int end) {
            return SGDSegment{update.parameter->mutable_data<float>(), update.gradient->data<float>(),
                update.velocity ? update.velocity->mutable_data<float>() : nullptr, count, begin, end, update.step};
        },
        [](const numeric::SGDUpdate& update, cudaStream_t stream) {
            const auto count = update.parameter->layout.numel();
            sgd_kernel<<<grid_size(count), block_size, 0, stream>>>(update.parameter->mutable_data<float>(),
                update.gradient->data<float>(), update.velocity ? update.velocity->mutable_data<float>() : nullptr,
                count, IndexMap(update.parameter->layout), IndexMap(update.gradient->layout),
                update.velocity ? IndexMap(update.velocity->layout) : IndexMap{}, update.step);
        });
}

void adam(std::span<const numeric::AdamUpdate> updates) {
    submit_optimizer_updates<AdamSegment>(updates,
        [](const auto& update) {
            return update.parameter->layout.is_contiguous() && update.gradient->layout.is_contiguous()
                && update.first->layout.is_contiguous() && update.second->layout.is_contiguous();
        },
        [](const numeric::AdamUpdate& update, Index count, int begin, int end) {
            return AdamSegment{update.parameter->mutable_data<float>(), update.gradient->data<float>(),
                update.first->mutable_data<float>(), update.second->mutable_data<float>(), count, begin, end, update.step};
        },
        [](const numeric::AdamUpdate& update, cudaStream_t stream) {
            const auto count = update.parameter->layout.numel();
            adam_kernel<<<grid_size(count), block_size, 0, stream>>>(update.parameter->mutable_data<float>(),
                update.gradient->data<float>(), update.first->mutable_data<float>(), update.second->mutable_data<float>(),
                count, IndexMap(update.parameter->layout), IndexMap(update.gradient->layout),
                IndexMap(update.first->layout), IndexMap(update.second->layout), update.step);
        });
}

void rmsprop(std::span<const numeric::RMSpropUpdate> updates) {
    submit_optimizer_updates<RMSpropSegment>(updates,
        [](const auto& update) {
            return update.parameter->layout.is_contiguous() && update.gradient->layout.is_contiguous()
                && update.square_average->layout.is_contiguous()
                && (!update.momentum || update.momentum->layout.is_contiguous())
                && (!update.gradient_average || update.gradient_average->layout.is_contiguous());
        },
        [](const numeric::RMSpropUpdate& update, Index count, int begin, int end) {
            return RMSpropSegment{update.parameter->mutable_data<float>(), update.gradient->data<float>(),
                update.square_average->mutable_data<float>(), update.momentum ? update.momentum->mutable_data<float>() : nullptr,
                update.gradient_average ? update.gradient_average->mutable_data<float>() : nullptr,
                count, begin, end, update.step};
        },
        [](const numeric::RMSpropUpdate& update, cudaStream_t stream) {
            const auto count = update.parameter->layout.numel();
            rmsprop_kernel<<<grid_size(count), block_size, 0, stream>>>(update.parameter->mutable_data<float>(),
                update.gradient->data<float>(), update.square_average->mutable_data<float>(),
                update.momentum ? update.momentum->mutable_data<float>() : nullptr,
                update.gradient_average ? update.gradient_average->mutable_data<float>() : nullptr, count,
                IndexMap(update.parameter->layout), IndexMap(update.gradient->layout), IndexMap(update.square_average->layout),
                update.momentum ? IndexMap(update.momentum->layout) : IndexMap{},
                update.gradient_average ? IndexMap(update.gradient_average->layout) : IndexMap{}, update.step);
        });
}

void adagrad(std::span<const numeric::AdagradUpdate> updates) {
    submit_optimizer_updates<AdagradSegment>(updates,
        [](const auto& update) {
            return update.parameter->layout.is_contiguous() && update.gradient->layout.is_contiguous()
                && update.sum->layout.is_contiguous();
        },
        [](const numeric::AdagradUpdate& update, Index count, int begin, int end) {
            return AdagradSegment{update.parameter->mutable_data<float>(), update.gradient->data<float>(),
                update.sum->mutable_data<float>(), count, begin, end, update.step};
        },
        [](const numeric::AdagradUpdate& update, cudaStream_t stream) {
            const auto count = update.parameter->layout.numel();
            adagrad_kernel<<<grid_size(count), block_size, 0, stream>>>(update.parameter->mutable_data<float>(),
                update.gradient->data<float>(), update.sum->mutable_data<float>(), count,
                IndexMap(update.parameter->layout), IndexMap(update.gradient->layout), IndexMap(update.sum->layout), update.step);
        });
}

} // namespace dl::detail::cuda

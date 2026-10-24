#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "core/dispatch.h"
#include "kernels/cuda/common.cuh"

namespace dl::detail::cuda {
namespace {

template <class T, numeric::Compare Operation>
__global__ void compare_kernel(const T* a, const T* b, bool* output, Index count,
                                IndexMap left, IndexMap right) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const T x = a[left.offset(index)];
        const T y = b[right.offset(index)];
        if constexpr (Operation == numeric::Compare::eq) output[index] = x == y;
        if constexpr (Operation == numeric::Compare::ne) output[index] = x != y;
        if constexpr (Operation == numeric::Compare::lt) output[index] = x < y;
        if constexpr (Operation == numeric::Compare::le) output[index] = x <= y;
        if constexpr (Operation == numeric::Compare::gt) output[index] = x > y;
        if constexpr (Operation == numeric::Compare::ge) output[index] = x >= y;
    }
}

} // namespace

TensorData compare(const TensorData& a, const TensorData& b, numeric::Compare operation) {
    auto shape = broadcast_shape(a.layout.sizes, b.layout.sizes);
    auto result = empty_data(shape, DType::boolean, a.device());
    const auto count = result.layout.numel();
    if (count != 0) {
        Execution execution(a);
        const IndexMap left(shape, broadcast_strides(a.layout, shape));
        const IndexMap right(shape, broadcast_strides(b.layout, shape));
        visit_dtype(a.dtype, [&]<class T> {
            auto launch = [&]<numeric::Compare Operation> {
                compare_kernel<T, Operation><<<grid_size(count), block_size, 0, execution.stream()>>>(
                    a.data<T>(), b.data<T>(), result.mutable_data<bool>(), count, left, right);
            };
            switch (operation) {
            case numeric::Compare::eq: return launch.template operator()<numeric::Compare::eq>();
            case numeric::Compare::ne: return launch.template operator()<numeric::Compare::ne>();
            case numeric::Compare::lt: return launch.template operator()<numeric::Compare::lt>();
            case numeric::Compare::le: return launch.template operator()<numeric::Compare::le>();
            case numeric::Compare::gt: return launch.template operator()<numeric::Compare::gt>();
            case numeric::Compare::ge: return launch.template operator()<numeric::Compare::ge>();
            }
        });
        check(cudaGetLastError());
    }
    return result;
}

} // namespace dl::detail::cuda

#include "kernels/cuda/ops.h"
#include "backends/cuda_context.h"

#include <cublasLt.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace dl::detail::cuda {
namespace {

// cuBLASLt 的描述符只持有库元数据，按作用域成对创建和销毁。
template <class T, auto Create, auto Destroy>
class BlasObject {
public:
    template <class... Args> explicit BlasObject(Args... args) { check(Create(&value_, args...)); }
    ~BlasObject() { Destroy(value_); }
    BlasObject(const BlasObject&) = delete;
    BlasObject& operator=(const BlasObject&) = delete;
    T get() const { return value_; }
private:
    T value_;
};

using Handle = BlasObject<cublasLtHandle_t, cublasLtCreate, cublasLtDestroy>;
using Operation = BlasObject<cublasLtMatmulDesc_t, cublasLtMatmulDescCreate, cublasLtMatmulDescDestroy>;
using Matrix = BlasObject<cublasLtMatrixLayout_t, cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy>;
using Preference = BlasObject<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceCreate, cublasLtMatmulPreferenceDestroy>;
using Key = std::tuple<DType, Index, Index, Index>;

constexpr std::size_t workspace_limit = 32 * 1024 * 1024;
constexpr std::size_t plan_limit = 64;
constexpr std::uint32_t alignment = 16;

cudaDataType_t storage_type(DType dtype) {
    return dtype == DType::f16 ? CUDA_R_16F : dtype == DType::bf16 ? CUDA_R_16BF : CUDA_R_32F;
}

struct Plan {
    Plan(cublasLtHandle_t handle, DType dtype, Index rows, Index columns, Index inner)
        : operation(CUBLAS_COMPUTE_32F, CUDA_R_32F),
          weight(storage_type(dtype), inner, columns, inner),
          input(storage_type(dtype), inner, rows, inner),
          bias(CUDA_R_32F, columns, rows, 0), output(CUDA_R_32F, columns, rows, columns) {
        // Y^T = W X^T + b：列主序视图共享原缓冲，C 的零列距把偏置广播到每个样本。
        const auto transpose = CUBLAS_OP_T;
        check(cublasLtMatmulDescSetAttribute(operation.get(), CUBLASLT_MATMUL_DESC_TRANSA,
                                             &transpose, sizeof(transpose)));
        Preference preference;
        check(cublasLtMatmulPreferenceSetAttribute(preference.get(), CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &workspace_limit, sizeof(workspace_limit)));
        for (auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                               CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
            check(cublasLtMatmulPreferenceSetAttribute(preference.get(), attribute, &alignment, sizeof(alignment)));
        int count = 0;
        const auto status = cublasLtMatmulAlgoGetHeuristic(handle, operation.get(), weight.get(), input.get(),
            bias.get(), output.get(), preference.get(), 1, &candidate, &count);
        if (status != CUBLAS_STATUS_NOT_SUPPORTED) check(status);
        supported = count != 0 && candidate.state == CUBLAS_STATUS_SUCCESS;
    }

    Operation operation;
    Matrix weight, input, bias, output;
    cublasLtMatmulHeuristicResult_t candidate{};
    bool supported = false;
};

bool packed(const TensorData& tensor) {
    return tensor.layout.is_contiguous()
        && reinterpret_cast<std::uintptr_t>(tensor.address()) % alignment == 0;
}

} // namespace

struct LinearContext {
    LinearContext(Device device, cudaStream_t stream, cudaMemPool_t pool)
        : stream(stream), pool(pool), workspace(nullptr, AsyncBufferDeleter{device.index, stream}) {}

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

    Handle handle;
    std::mutex mutex;
    std::map<Key, std::unique_ptr<Plan>> plans;
    cudaStream_t stream;
    cudaMemPool_t pool;
    // 工作区借用上下文的 stream/pool，释放顺序与普通 CUDA 分配一致。
    std::unique_ptr<void, AsyncBufferDeleter> workspace;
    std::size_t capacity = 0;
};

std::shared_ptr<LinearContext> make_linear_context(Device device, cudaStream_t stream, cudaMemPool_t pool) {
    return std::make_shared<LinearContext>(device, stream, pool);
}

void release_linear_workspace(LinearContext& context) {
    const std::lock_guard lock(context.mutex);
    context.workspace.reset();
    context.capacity = 0;
}

bool linear_into(TensorData& result, const TensorData& input, const TensorData& weight, const TensorData& bias) {
    const auto rows = input.layout.sizes[0], inner = input.layout.sizes[1], columns = weight.layout.sizes[0];
    if (result.layout.numel() == 0 || inner == 0 || bias.layout.sizes.size() != 1 || bias.layout.sizes[0] != columns
        || !packed(input) || !packed(weight) || !packed(bias)) return false;
    Execution execution(input);
    auto& context = *execution.context.linear;
    const std::lock_guard lock(context.mutex);
    const Key key{input.dtype, rows, columns, inner};
    auto found = context.plans.find(key);
    if (found == context.plans.end()) {
        if (context.plans.size() == plan_limit) context.plans.clear();
        found = context.plans.emplace(key, std::make_unique<Plan>(context.handle.get(), input.dtype,
                                                                 rows, columns, inner)).first;
    }
    const auto& plan = *found->second;
    if (!plan.supported) return false;
    const float alpha = 1.0f, beta = 1.0f;
    check(cublasLtMatmul(context.handle.get(), plan.operation.get(), &alpha,
        weight.address(), plan.weight.get(), input.address(), plan.input.get(), &beta,
        bias.address(), plan.bias.get(), result.mutable_address(), plan.output.get(), &plan.candidate.algo,
        context.workspace_for(plan.candidate.workspaceSize), plan.candidate.workspaceSize, execution.stream()));
    return true;
}

} // namespace dl::detail::cuda

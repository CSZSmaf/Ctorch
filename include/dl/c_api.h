#ifndef CTORCH_C_API_H
#define CTORCH_C_API_H

#include "dl/export.h"
#include "dl/version.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dl_tensor dl_tensor;
typedef struct dl_tensor_list dl_tensor_list;
typedef struct dl_gradients dl_gradients;
typedef struct dl_module dl_module;
typedef struct dl_optimizer dl_optimizer;
typedef struct dl_scheduler dl_scheduler;
typedef struct dl_generator dl_generator;
typedef struct dl_loader dl_loader;
typedef struct dl_prefetch_loader dl_prefetch_loader;
typedef struct dl_transfer dl_transfer;
typedef struct dl_state dl_state;
typedef struct dl_checkpoint dl_checkpoint;
typedef struct dl_functional dl_functional;
typedef struct dl_functional_state dl_functional_state;
typedef struct dl_grad_scope dl_grad_scope;
typedef struct dl_autocast_scope dl_autocast_scope;
typedef struct dl_grad_scaler dl_grad_scaler;
typedef struct dl_vjp_context dl_vjp_context;

typedef enum dl_status { DL_OK = 0, DL_ERROR = 1 } dl_status;
typedef enum dl_dtype { DL_F32 = 0, DL_F64 = 1, DL_I64 = 2, DL_BOOL = 3, DL_F16 = 4, DL_BF16 = 5 } dl_dtype;
typedef enum dl_device_type { DL_CPU = 0, DL_CUDA = 1 } dl_device_type;
typedef enum dl_loss_reduction { DL_REDUCTION_NONE = 0, DL_REDUCTION_MEAN = 1, DL_REDUCTION_SUM = 2 } dl_loss_reduction;
typedef enum dl_interpolation_mode {
    DL_NEAREST = 0, DL_NEAREST_EXACT = 1, DL_BILINEAR = 2
} dl_interpolation_mode;
typedef enum dl_rnn_activation { DL_RNN_TANH = 0, DL_RNN_RELU = 1 } dl_rnn_activation;

typedef struct dl_device { dl_device_type type; int32_t index; } dl_device;

/* Ctorch 专用 CUDA 池的 Runtime 字节统计；不包含外部导入和设备库自身分配。 */
typedef struct dl_cuda_memory_stats {
    uint64_t used_bytes, reserved_bytes;
    uint64_t peak_used_bytes, peak_reserved_bytes;
    uint64_t release_threshold;
} dl_cuda_memory_stats;
typedef struct dl_shape { const int64_t* data; size_t size; } dl_shape;
typedef struct dl_size2 { int64_t height, width; } dl_size2;

typedef struct dl_tensor_options {
    dl_dtype dtype;
    dl_device device;
    int32_t requires_grad;
} dl_tensor_options;

typedef struct dl_grad_scaler_options {
    double initial_scale, growth_factor, backoff_factor;
    int64_t growth_interval;
    int32_t enabled;
} dl_grad_scaler_options;

typedef struct dl_grad_scaler_state {
    dl_grad_scaler_options options;
    double scale;
    int64_t finite_steps;
} dl_grad_scaler_state;

typedef struct dl_tensor_info {
    dl_dtype dtype;
    dl_device device;
    int64_t numel;
    int32_t contiguous, requires_grad, leaf;
} dl_tensor_info;

/* shape/strides 借用所属张量或回调上下文；偏移以元素计，data 地址已经包含该偏移。 */
typedef struct dl_tensor_spec {
    dl_shape shape, strides;
    dl_dtype dtype;
    dl_device device;
    int64_t storage_offset;
    int32_t read_only;
} dl_tensor_spec;

/* 采样按值读取 key，不推进 Generator；stream 选择 Philox counter 的高 64 位。 */
typedef struct dl_random_key { uint64_t seed, stream; } dl_random_key;

/* 回调借用输入与上下文，将拥有型结果句柄交给框架；同一结果用于多槽时分别 retain。
 * 失败时返回 DL_ERROR，error_message 可引用静态文本、user_data 文本或 dl_last_error()。
 * 前向结果有一个槽，反向槽数等于输入数，初始为 NULL；NULL 梯度表示没有贡献。 */
typedef dl_status (*dl_vjp_forward_fn)(dl_vjp_context* context, const dl_tensor* const* inputs,
    size_t count, void* user_data, dl_tensor** output, const char** error_message);
typedef dl_status (*dl_vjp_backward_fn)(const dl_vjp_context* context, const dl_tensor* grad,
    void* user_data, dl_tensor** gradients, size_t count, const char** error_message);
typedef void (*dl_user_data_deleter)(void* user_data);

/* 激活重算回调在前向和需要梯度时分别执行，输入句柄只在本次回调内借用。
 * 每次交付一个拥有型输出句柄；错误文本和句柄交接规则沿用 custom VJP。 */
typedef dl_status (*dl_recompute_fn)(const dl_tensor* const* inputs, size_t count,
    void* user_data, dl_tensor** output, const char** error_message);
typedef struct dl_checkpoint_options { int32_t preserve_rng_state; } dl_checkpoint_options;

typedef struct dl_conv2d_options {
    dl_size2 stride, padding, dilation;
    int64_t groups;
} dl_conv2d_options;

typedef struct dl_pool2d_options { dl_size2 kernel, stride, padding; } dl_pool2d_options;
typedef struct dl_conv1d_options { int64_t stride, padding, dilation, groups; } dl_conv1d_options;
typedef struct dl_pool1d_options { int64_t kernel, stride, padding; } dl_pool1d_options;
typedef struct dl_batch_norm_options {
    double epsilon, momentum;
    int32_t affine, track_running_stats;
} dl_batch_norm_options;
typedef struct dl_attention_options { double dropout; int32_t bias; } dl_attention_options;
typedef enum dl_attention_backend { DL_ATTENTION_MATH = 0, DL_ATTENTION_CUDNN = 1 } dl_attention_backend;
typedef struct dl_attention_compute_options {
    double dropout;
    int32_t causal;
    dl_attention_backend backend;
} dl_attention_compute_options;
typedef struct dl_recurrent_options {
    int64_t layers;
    int32_t bias, batch_first, bidirectional;
    double dropout;
} dl_recurrent_options;
typedef struct dl_sgd_options {
    double learning_rate, momentum, dampening, weight_decay;
    int32_t nesterov;
} dl_sgd_options;
typedef struct dl_adamw_options { double learning_rate, beta1, beta2, epsilon, weight_decay; } dl_adamw_options;
typedef struct dl_adam_options { double learning_rate, beta1, beta2, epsilon, weight_decay; } dl_adam_options;
typedef struct dl_rmsprop_options {
    double learning_rate, alpha, epsilon, weight_decay, momentum;
    int32_t centered;
} dl_rmsprop_options;
typedef struct dl_adagrad_options {
    double learning_rate, learning_rate_decay, weight_decay, initial_accumulator, epsilon;
} dl_adagrad_options;
/* 配置指针只在创建期间借用；NULL 使用优化器默认值。 */
typedef struct dl_parameter_group {
    const dl_tensor* const* parameters;
    size_t count;
    const double* learning_rate;
    const double* weight_decay;
} dl_parameter_group;
typedef struct dl_loader_options { int64_t batch_size; int32_t shuffle, drop_last; } dl_loader_options;
typedef struct dl_training_progress { int64_t epoch, step; } dl_training_progress;
typedef struct dl_named_generator { const char* name; dl_generator* generator; } dl_named_generator;

/* 创建和计算返回的新句柄由调用方 release；复制句柄共享底层对象。
 * status 入口在失败时返回 DL_ERROR，新建句柄的输出槽置为 NULL，实际失败文本保留在当前线程。
 * dl_last_error 的地址有效到本线程下一次 status 操作；元数据查询不改变该文本。
 * 形状、dtype、设备、非负 stride 与别名约定遵循 C++ API，名称和路径使用 UTF-8。
 * 输入输出的句柄槽分别提供，release 接受 NULL；DL_BOOL 的宿主元素使用 bool。 */
DL_API const char* dl_version(void);
DL_API const char* dl_last_error(void);
DL_API dl_tensor_options dl_default_tensor_options(void);
DL_API dl_conv2d_options dl_default_conv2d_options(void);
DL_API dl_conv1d_options dl_default_conv1d_options(void);
DL_API dl_pool1d_options dl_default_pool1d_options(void);
DL_API dl_pool2d_options dl_default_pool2d_options(void);
DL_API dl_batch_norm_options dl_default_batch_norm_options(void);
DL_API dl_attention_options dl_default_attention_options(void);
DL_API dl_attention_compute_options dl_default_attention_compute_options(void);
DL_API dl_recurrent_options dl_default_recurrent_options(void);
DL_API dl_sgd_options dl_default_sgd_options(void);
DL_API dl_adamw_options dl_default_adamw_options(void);
DL_API dl_adam_options dl_default_adam_options(void);
DL_API dl_rmsprop_options dl_default_rmsprop_options(void);
DL_API dl_adagrad_options dl_default_adagrad_options(void);
DL_API dl_loader_options dl_default_loader_options(void);

DL_API void dl_tensor_release(dl_tensor* tensor);
DL_API dl_status dl_tensor_retain(const dl_tensor* tensor, dl_tensor** output);
DL_API dl_tensor_info dl_tensor_get_info(const dl_tensor* tensor);
DL_API dl_tensor_spec dl_tensor_get_spec(const dl_tensor* tensor);
/* shape、stride 和 data 借用张量对象，生命周期延续至对象释放或模型设备迁移。 */
DL_API dl_shape dl_tensor_shape(const dl_tensor* tensor);
DL_API dl_shape dl_tensor_strides(const dl_tensor* tensor);
DL_API const void* dl_tensor_data(const dl_tensor* tensor);
DL_API size_t dl_tensor_nbytes(const dl_tensor* tensor);
/* destination 提供 nbytes 字节的宿主空间，内容按逻辑行主序复制。 */
DL_API dl_status dl_tensor_copy_to_host(const dl_tensor* tensor, void* destination);
DL_API dl_status dl_from_host(const void* data, dl_shape shape, dl_tensor_options options, dl_tensor** output);
DL_API dl_status dl_full(dl_shape shape, double value, dl_tensor_options options, dl_tensor** output);
DL_API dl_status dl_zeros(dl_shape shape, dl_tensor_options options, dl_tensor** output);
DL_API dl_status dl_ones(dl_shape shape, dl_tensor_options options, dl_tensor** output);
/* like 工厂继承 shape/dtype/device，数据独立；requires_grad 单独指定。 */
DL_API dl_status dl_full_like(const dl_tensor* input, double value, int32_t requires_grad, dl_tensor** output);
DL_API dl_status dl_zeros_like(const dl_tensor* input, int32_t requires_grad, dl_tensor** output);
DL_API dl_status dl_ones_like(const dl_tensor* input, int32_t requires_grad, dl_tensor** output);
DL_API dl_status dl_arange(double start, double end, double step, dl_tensor_options options, dl_tensor** output);
DL_API dl_status dl_linspace(double start, double end, int64_t steps, dl_tensor_options options, dl_tensor** output);
DL_API dl_status dl_eye(int64_t rows, int64_t columns, dl_tensor_options options, dl_tensor** output);

DL_API dl_status dl_clone(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_detach(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_contiguous(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_astype(const dl_tensor* input, dl_dtype dtype, dl_tensor** output);
DL_API dl_status dl_to(const dl_tensor* input, dl_device device, dl_tensor** output);
DL_API dl_status dl_view(const dl_tensor* input, dl_shape shape, dl_tensor** output);
DL_API dl_status dl_reshape(const dl_tensor* input, dl_shape shape, dl_tensor** output);
/* 合并 first 到 last 的轴区间，负轴从末尾计数；标量返回长度一的视图。 */
DL_API dl_status dl_flatten(const dl_tensor* input, int64_t first, int64_t last, dl_tensor** output);
DL_API dl_status dl_transpose(const dl_tensor* input, int64_t first, int64_t second, dl_tensor** output);
DL_API dl_status dl_permute(const dl_tensor* input, dl_shape axes, dl_tensor** output);
DL_API dl_status dl_slice(const dl_tensor* input, int64_t axis, int64_t start, int64_t stop, int64_t step,
    dl_tensor** output);
DL_API dl_status dl_expand(const dl_tensor* input, dl_shape shape, dl_tensor** output);
DL_API dl_status dl_unsqueeze(const dl_tensor* input, int64_t axis, dl_tensor** output);
DL_API dl_status dl_squeeze(const dl_tensor* input, int64_t axis, dl_tensor** output);
DL_API dl_status dl_squeeze_all(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_select(const dl_tensor* input, int64_t axis, int64_t index, dl_tensor** output);
DL_API dl_status dl_narrow(const dl_tensor* input, int64_t axis, int64_t start, int64_t length, dl_tensor** output);
DL_API dl_status dl_repeat(const dl_tensor* input, dl_shape repeats, dl_tensor** output);
/* create 共享输入张量身份，输入句柄仍归调用方；count 为零时 tensors 可为 NULL。
 * get 返回独立拥有型句柄，释放列表后已取出的张量继续有效。 */
DL_API dl_status dl_tensor_list_create(const dl_tensor* const* tensors, size_t count, dl_tensor_list** output);
DL_API void dl_tensor_list_release(dl_tensor_list* list);
DL_API size_t dl_tensor_list_size(const dl_tensor_list* list);
DL_API dl_status dl_tensor_list_get(const dl_tensor_list* list, size_t index, dl_tensor** output);
DL_API dl_status dl_split(const dl_tensor* input, int64_t split_size, int64_t axis, dl_tensor_list** output);
DL_API dl_status dl_split_with_sizes(const dl_tensor* input, dl_shape sizes, int64_t axis, dl_tensor_list** output);
DL_API dl_status dl_chunk(const dl_tensor* input, int64_t chunks, int64_t axis, dl_tensor_list** output);
DL_API dl_status dl_unbind(const dl_tensor* input, int64_t axis, dl_tensor_list** output);

DL_API dl_status dl_add(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
/* 广播加法与 ReLU 共用一次遍历，结果为独立拥有的句柄，沿两侧输入建立求导关系。 */
DL_API dl_status dl_add_relu(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
DL_API dl_status dl_sub(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
DL_API dl_status dl_mul(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
DL_API dl_status dl_div(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
DL_API dl_status dl_pow(const dl_tensor* base, const dl_tensor* exponent, dl_tensor** output);
DL_API dl_status dl_pow_scalar(const dl_tensor* base, double exponent, dl_tensor** output);
DL_API dl_status dl_scalar_pow(double base, const dl_tensor* exponent, dl_tensor** output);
/* 比较按输入 dtype 和广播布局求值，返回不参与求导的 bool 张量。 */
DL_API dl_status dl_eq(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
DL_API dl_status dl_ne(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
DL_API dl_status dl_lt(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
DL_API dl_status dl_le(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
DL_API dl_status dl_gt(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
DL_API dl_status dl_ge(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
DL_API dl_status dl_add_scalar(const dl_tensor* input, double value, dl_tensor** output);
DL_API dl_status dl_mul_scalar(const dl_tensor* input, double value, dl_tensor** output);
DL_API dl_status dl_neg(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_exp(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_log(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_log1p(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_expm1(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_abs(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_sin(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_cos(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_tan(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_asin(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_acos(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_atan(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_clamp(const dl_tensor* input, double minimum, double maximum, dl_tensor** output);
DL_API dl_status dl_clamp_min(const dl_tensor* input, double minimum, dl_tensor** output);
DL_API dl_status dl_clamp_max(const dl_tensor* input, double maximum, dl_tensor** output);
DL_API dl_status dl_sqrt(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_tanh(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_sigmoid(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_relu(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_gelu(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_softplus(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_silu(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_leaky_relu(const dl_tensor* input, double negative_slope, dl_tensor** output);
/* axes 为空时复制逻辑数据并直接传递梯度，保留符号零；带 _all 的入口归约全部轴。 */
DL_API dl_status dl_sum(const dl_tensor* input, dl_shape axes, int32_t keepdim, dl_tensor** output);
DL_API dl_status dl_mean(const dl_tensor* input, dl_shape axes, int32_t keepdim, dl_tensor** output);
DL_API dl_status dl_sum_all(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_mean_all(const dl_tensor* input, dl_tensor** output);
/* amax 的归约组非空，并列最大位置均分梯度，NaN 随最大值传播。 */
DL_API dl_status dl_amax(const dl_tensor* input, dl_shape axes, int32_t keepdim, dl_tensor** output);
DL_API dl_status dl_amax_all(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_amin(const dl_tensor* input, dl_shape axes, int32_t keepdim, dl_tensor** output);
DL_API dl_status dl_amin_all(const dl_tensor* input, dl_tensor** output);
/* 非空组的首个极值位置，结果为 int64；_all 按逻辑行优先顺序编号。 */
DL_API dl_status dl_argmax(const dl_tensor* input, int64_t axis, int32_t keepdim, dl_tensor** output);
DL_API dl_status dl_argmax_all(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_argmin(const dl_tensor* input, int64_t axis, int32_t keepdim, dl_tensor** output);
DL_API dl_status dl_argmin_all(const dl_tensor* input, dl_tensor** output);
DL_API dl_status dl_logsumexp(const dl_tensor* input, dl_shape axes, int32_t keepdim, dl_tensor** output);
DL_API dl_status dl_logsumexp_all(const dl_tensor* input, dl_tensor** output);
/* 方差分母为 N-correction，N > correction >= 0；标准差的常数组梯度为零。 */
DL_API dl_status dl_var(const dl_tensor* input, dl_shape axes, double correction, int32_t keepdim, dl_tensor** output);
DL_API dl_status dl_var_all(const dl_tensor* input, double correction, dl_tensor** output);
DL_API dl_status dl_stddev(const dl_tensor* input, dl_shape axes, double correction, int32_t keepdim, dl_tensor** output);
DL_API dl_status dl_stddev_all(const dl_tensor* input, double correction, dl_tensor** output);
DL_API dl_status dl_vector_norm(const dl_tensor* input, dl_shape axes, double order, int32_t keepdim, dl_tensor** output);
DL_API dl_status dl_vector_norm_all(const dl_tensor* input, double order, dl_tensor** output);
/* 向量临时补入矩阵维，结果移除对应维；两个向量返回标量，批维按广播组合。 */
DL_API dl_status dl_matmul(const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
/* input [..., in] 与 weight [out, in] 计算末尾特征投影，bias [out] 可为 NULL。 */
DL_API dl_status dl_linear(const dl_tensor* input, const dl_tensor* weight, const dl_tensor* bias,
    dl_tensor** output);
DL_API dl_status dl_softmax(const dl_tensor* input, int64_t axis, dl_tensor** output);
DL_API dl_status dl_log_softmax(const dl_tensor* input, int64_t axis, dl_tensor** output);
DL_API dl_status dl_where(const dl_tensor* mask, const dl_tensor* a, const dl_tensor* b, dl_tensor** output);
DL_API dl_status dl_index_select(const dl_tensor* input, int64_t axis, const dl_tensor* indices, dl_tensor** output);
/* 任意形状的 int64 indices 从二维浮点 weight 读取行，重复索引累计稠密梯度。 */
DL_API dl_status dl_embedding(const dl_tensor* indices, const dl_tensor* weight, dl_tensor** output);
/* gather 的结果形状等于 indices；scatter_add 在 input 的副本中累计 source 前缀。 */
DL_API dl_status dl_gather(const dl_tensor* input, int64_t axis, const dl_tensor* indices, dl_tensor** output);
DL_API dl_status dl_scatter_add(const dl_tensor* input, int64_t axis, const dl_tensor* indices,
    const dl_tensor* source, dl_tensor** output);
/* 非空输入列表的句柄由调用方继续持有，结果拥有新的连续存储。 */
DL_API dl_status dl_concat(const dl_tensor* const* inputs, size_t count, int64_t axis, dl_tensor** output);
DL_API dl_status dl_stack(const dl_tensor* const* inputs, size_t count, int64_t axis, dl_tensor** output);
DL_API dl_status dl_cross_entropy(const dl_tensor* logits, const dl_tensor* labels, dl_tensor** output);
DL_API dl_status dl_mse_loss(const dl_tensor* prediction, const dl_tensor* target, dl_tensor** output);
/* none 保留逐项损失，mean 和 sum 返回标量；原有两项损失入口使用 mean。 */
DL_API dl_status dl_cross_entropy_with_reduction(const dl_tensor* logits, const dl_tensor* labels,
    dl_loss_reduction reduction, dl_tensor** output);
/// ignore_index 对应的位置不参与损失及梯度，mean 按有效标签数量归约。
DL_API dl_status dl_cross_entropy_ignore_index(const dl_tensor* logits, const dl_tensor* labels,
    dl_loss_reduction reduction, int64_t ignore_index, dl_tensor** output);
DL_API dl_status dl_mse_loss_with_reduction(const dl_tensor* prediction, const dl_tensor* target,
    dl_loss_reduction reduction, dl_tensor** output);
DL_API dl_status dl_nll_loss(const dl_tensor* log_probabilities, const dl_tensor* labels,
    dl_loss_reduction reduction, dl_tensor** output);
DL_API dl_status dl_nll_loss_ignore_index(const dl_tensor* log_probabilities, const dl_tensor* labels,
    dl_loss_reduction reduction, int64_t ignore_index, dl_tensor** output);
DL_API dl_status dl_l1_loss(const dl_tensor* prediction, const dl_tensor* target,
    dl_loss_reduction reduction, dl_tensor** output);
DL_API dl_status dl_smooth_l1_loss(const dl_tensor* prediction, const dl_tensor* target,
    double beta, dl_loss_reduction reduction, dl_tensor** output);
DL_API dl_status dl_huber_loss(const dl_tensor* prediction, const dl_tensor* target,
    double delta, dl_loss_reduction reduction, dl_tensor** output);
DL_API dl_status dl_binary_cross_entropy_with_logits(const dl_tensor* logits, const dl_tensor* target,
    dl_loss_reduction reduction, dl_tensor** output);
DL_API dl_status dl_binary_cross_entropy(const dl_tensor* probability, const dl_tensor* target,
    dl_loss_reduction reduction, dl_tensor** output);
DL_API dl_status dl_conv2d(const dl_tensor* input, const dl_tensor* weight, const dl_tensor* bias,
    dl_conv2d_options options, dl_tensor** output);
DL_API dl_status dl_max_pool2d(const dl_tensor* input, dl_pool2d_options options, dl_tensor** output);
DL_API dl_status dl_avg_pool2d(const dl_tensor* input, dl_pool2d_options options, int32_t count_include_pad,
    dl_tensor** output);
/* 末尾 normalized_shape 维使用总体方差；weight 和 bias 分别可为 NULL。 */
DL_API dl_status dl_layer_norm(const dl_tensor* input, dl_shape normalized_shape,
    const dl_tensor* weight, const dl_tensor* bias, double epsilon, dl_tensor** output);
/* running_mean/variance 同时提供或同时为 NULL，weight/bias 分别可为 NULL。
   返回列表依次为 output、next_mean、next_variance；无运行状态时列表仅含 output。
   状态为 [C]、累积 dtype，训练返回独立新状态，eval 返回脱离求导历史的存储别名。 */
DL_API dl_status dl_batch_norm(const dl_tensor* input, const dl_tensor* running_mean,
    const dl_tensor* running_variance, const dl_tensor* weight, const dl_tensor* bias,
    int32_t training, double momentum, double epsilon, dl_tensor_list** output);
/* 提供的两份状态存储可写且互不重叠，与输入/仿射参数分开；训练原地更新，eval 只读。 */
DL_API dl_status dl_batch_norm_update(const dl_tensor* input, const dl_tensor* running_mean,
    const dl_tensor* running_variance, const dl_tensor* weight, const dl_tensor* bias,
    int32_t training, double momentum, double epsilon, dl_tensor** output);
DL_API dl_status dl_group_norm(const dl_tensor* input, int64_t groups, const dl_tensor* weight,
    const dl_tensor* bias, double epsilon, dl_tensor** output);
/* RMSNorm 的 epsilon 为 NULL 时使用 dtype 机器精度；非空指针只在调用期间读取。 */
DL_API dl_status dl_rms_norm(const dl_tensor* input, dl_shape normalized_shape,
    const dl_tensor* weight, const double* epsilon, dl_tensor** output);
DL_API dl_status dl_conv1d(const dl_tensor* input, const dl_tensor* weight, const dl_tensor* bias,
    dl_conv1d_options options, dl_tensor** output);
DL_API dl_status dl_max_pool1d(const dl_tensor* input, dl_pool1d_options options, dl_tensor** output);
DL_API dl_status dl_avg_pool1d(const dl_tensor* input, dl_pool1d_options options, int32_t count_include_pad, dl_tensor** output);
DL_API dl_status dl_adaptive_avg_pool1d(const dl_tensor* input, int64_t output_size, dl_tensor** output);
DL_API dl_status dl_adaptive_avg_pool2d(const dl_tensor* input, dl_size2 output_size, dl_tensor** output);
DL_API dl_status dl_interpolate(const dl_tensor* input, dl_size2 output_size, dl_interpolation_mode mode,
    int32_t align_corners, dl_tensor** output);
DL_API dl_status dl_dropout(const dl_tensor* input, double probability, int32_t training, dl_tensor** output);
/* generator 仅在调用期间借用，为 NULL 时使用当前线程的默认流。 */
DL_API dl_status dl_dropout_with_generator(const dl_tensor* input, double probability, int32_t training,
    dl_generator* generator, dl_tensor** output);
DL_API dl_status dl_dropout_with_key(const dl_tensor* input, double probability, int32_t training,
    dl_random_key key, dl_tensor** output);
DL_API dl_status dl_scaled_dot_product_attention(const dl_tensor* query, const dl_tensor* key,
    const dl_tensor* value, const dl_tensor* keep_mask, double dropout, dl_tensor** output);
/* random_key 为 NULL 时使用默认随机流；显式 key 按值借用，后端与因果计算由 options 选择。 */
DL_API dl_status dl_scaled_dot_product_attention_ex(const dl_tensor* query, const dl_tensor* key,
    const dl_tensor* value, const dl_tensor* keep_mask, dl_attention_compute_options options,
    const dl_random_key* random_key, dl_tensor** output);
DL_API dl_status dl_causal_mask(int64_t query_length, int64_t key_length, dl_device device, dl_tensor** output);
/* 单步门顺序：GRU 为 reset/update/new，LSTM 为 input/forget/candidate/output。偏置可为 NULL。 */
DL_API dl_status dl_rnn_cell(const dl_tensor* input, const dl_tensor* hidden,
    const dl_tensor* weight_ih, const dl_tensor* weight_hh, const dl_tensor* bias_ih, const dl_tensor* bias_hh,
    dl_rnn_activation activation, dl_tensor** output);
DL_API dl_status dl_gru_cell(const dl_tensor* input, const dl_tensor* hidden,
    const dl_tensor* weight_ih, const dl_tensor* weight_hh, const dl_tensor* bias_ih, const dl_tensor* bias_hh,
    dl_tensor** output);
/* LSTM 单步列表依次为 hidden、cell，列表项沿用 dl_tensor_list_get 的拥有型句柄规则。 */
DL_API dl_status dl_lstm_cell(const dl_tensor* input, const dl_tensor* hidden, const dl_tensor* cell,
    const dl_tensor* weight_ih, const dl_tensor* weight_hh, const dl_tensor* bias_ih, const dl_tensor* bias_hh,
    dl_tensor_list** output);

/* seed 为 NULL 时对标量输出使用种子 1；未建立叶梯度时 gradient 返回 NULL。 */
DL_API dl_status dl_backward(const dl_tensor* output, const dl_tensor* seed);
DL_API dl_status dl_tensor_gradient(const dl_tensor* tensor, dl_tensor** output);
DL_API dl_status dl_tensor_zero_grad(const dl_tensor* tensor);
/* 梯度值按 inputs 顺序交付，叶梯度槽保持原样；seed 为 NULL 时 output 为标量。
   输入可为叶或中间结果，冻结和未收到贡献的项保留为空；一次调用消费可达前向图。 */
DL_API dl_status dl_grad(const dl_tensor* output, const dl_tensor* const* inputs,
    size_t input_count, const dl_tensor* seed, dl_gradients** gradients);
/* 多输出 VJP 求和；seeds 为 NULL 时所有输出均为标量，否则提供 output_count 个种子。
   每个种子与对应输出同 shape、设备，dtype 为输出 dtype 或其累积 dtype。 */
DL_API dl_status dl_grad_many(const dl_tensor* const* outputs, size_t output_count,
    const dl_tensor* const* inputs, size_t input_count, const dl_tensor* const* seeds,
    dl_gradients** gradients);
/* 从可空句柄数组组成梯度列表，保留所传张量身份及重复关系。 */
DL_API dl_status dl_gradients_create(const dl_tensor* const* values, size_t count, dl_gradients** output);
DL_API void dl_gradients_release(dl_gradients* gradients);
DL_API size_t dl_gradients_size(const dl_gradients* gradients);
/* 有值时交付拥有型张量句柄，可独立于列表存活；空项交付 NULL，状态仍为 DL_OK。
   dl_grad 产生的重复输入结果共享梯度身份，其值不带求导历史。 */
DL_API dl_status dl_gradients_get(const dl_gradients* gradients, size_t index, dl_tensor** output);
/* 浮点叶张量在完整前向图之间切换状态；关闭清空梯度，重新开启建立空梯度槽。 */
DL_API dl_status dl_tensor_set_requires_grad(const dl_tensor* tensor, int32_t enabled);
/* NoGrad 作用域在同一线程按后进先出结束，恢复进入前的图记录状态。 */
DL_API dl_status dl_no_grad_begin(dl_grad_scope** output);
DL_API void dl_no_grad_end(dl_grad_scope* scope);
/* 显式开启或关闭记录；可在 NoGrad 内临时开启，按同一线程的嵌套顺序结束。 */
DL_API dl_status dl_grad_mode_begin(int32_t enabled, dl_grad_scope** output);
DL_API void dl_grad_mode_end(dl_grad_scope* scope);

/* 自定义 VJP 的前反向均在 NoGrad 下执行；反向恢复本次前向入口的 autocast 设置。
 * 调用接管 user_data 的释放责任，deleter 在图消费或释放时调用一次，无图时在调用结束前释放。
 * deleter 为 NULL 时由调用方保持 user_data 存活；回调函数代码在图存活期间保持可调用。 */
DL_API dl_status dl_custom_vjp(const dl_tensor* const* inputs, size_t count,
    dl_vjp_forward_fn forward, dl_vjp_backward_fn backward, void* user_data,
    dl_user_data_deleter deleter, dl_tensor** output);
DL_API size_t dl_vjp_input_count(const dl_vjp_context* context);
DL_API int32_t dl_vjp_needs_input_grad(const dl_vjp_context* context, size_t index);
DL_API dl_tensor_spec dl_vjp_input_spec(const dl_vjp_context* context, size_t index);
/* 保存值只在前向设置；读取返回独立拥有型句柄，释放回调上下文后仍可使用。 */
DL_API dl_status dl_vjp_save_for_backward(dl_vjp_context* context, const dl_tensor* const* tensors, size_t count);
DL_API size_t dl_vjp_saved_count(const dl_vjp_context* context);
DL_API dl_status dl_vjp_saved_tensor(const dl_vjp_context* context, size_t index, dl_tensor** output);

/* 与文件训练快照独立；保存输入并在反向时重算 function，参数也作为显式输入传入。
 * function 只读输入和固定配置，外部 buffer 更新在块外完成；默认保留线程默认随机流。
 * 调用开始接管 user_data，最后一个重算闭包释放时执行可选 deleter。 */
DL_API dl_checkpoint_options dl_default_checkpoint_options(void);
DL_API dl_status dl_activation_checkpoint(const dl_tensor* const* inputs, size_t count,
    dl_recompute_fn function, void* user_data, dl_user_data_deleter deleter,
    dl_checkpoint_options options, dl_tensor** output);

/* 当前线程按作用域设置 GEMM 与卷积精度，dtype 为 DL_F16 或 DL_BF16；按嵌套顺序结束。 */
DL_API dl_status dl_autocast_begin(dl_device_type device, dl_dtype dtype, int32_t enabled, dl_autocast_scope** output);
DL_API void dl_autocast_end(dl_autocast_scope* scope);

/* 同一更新窗口保持倍率不变；状态在完成 step 或 update 后保存。 */
DL_API dl_grad_scaler_options dl_default_grad_scaler_options(void);
DL_API dl_status dl_grad_scaler_create(dl_grad_scaler_options options, dl_grad_scaler** output);
DL_API void dl_grad_scaler_release(dl_grad_scaler* scaler);
DL_API double dl_grad_scaler_get_scale(const dl_grad_scaler* scaler);
DL_API dl_status dl_grad_scaler_scale(const dl_grad_scaler* scaler, const dl_tensor* loss, dl_tensor** output);
/* unscale 原地还原梯度并返回有限标志；显式裁剪流程随后条件调用 optimizer_step 和 update。 */
DL_API dl_status dl_grad_scaler_unscale(const dl_grad_scaler* scaler, dl_optimizer* optimizer, int32_t* finite);
/* 独立返回还原结果和有限标志，原梯度只读；空项与重复身份关系保持原样。 */
DL_API dl_status dl_grad_scaler_unscale_gradients(const dl_grad_scaler* scaler,
    const dl_gradients* gradients, dl_gradients** output, int32_t* finite);
/* 原地处理连续可写梯度，同身份只处理一次，不同身份的逻辑元素彼此不重叠。
   CPU 为 f32/f64，CUDA 为 f32；禁用时不扫描并返回 true，倍率由后续 update 调整。 */
DL_API dl_status dl_grad_scaler_unscale_gradients_update(const dl_grad_scaler* scaler,
    dl_gradients* gradients, int32_t* finite);
DL_API dl_status dl_grad_scaler_update(dl_grad_scaler* scaler, int32_t finite);
/* step 组合还原、条件优化器更新和倍率更新，updated 表示是否调用优化器。 */
DL_API dl_status dl_grad_scaler_step(dl_grad_scaler* scaler, dl_optimizer* optimizer, int32_t* updated);
DL_API dl_status dl_grad_scaler_get_state(const dl_grad_scaler* scaler, dl_grad_scaler_state* state);
DL_API dl_status dl_grad_scaler_load_state(dl_grad_scaler* scaler, const dl_grad_scaler_state* state);
DL_API int32_t dl_grad_enabled(void);
DL_API dl_status dl_synchronize(dl_device device);

/* 查询与峰值重置不等待 stream；需要阶段统计时先 dl_synchronize。
 * CUDA 构建按设备惰性创建池，CPU-only 构建通过 DL_ERROR 交付功能不可用。 */
DL_API dl_status dl_cuda_memory_get_stats(int32_t device, dl_cuda_memory_stats* output);
/* 阈值在同步点影响空闲内存归还，默认零，不表示显存容量上限。 */
DL_API dl_status dl_cuda_memory_set_release_threshold(int32_t device, uint64_t bytes);
DL_API dl_status dl_cuda_memory_reset_peak_stats(int32_t device);
/* 在训练阶段之间归还工作区、等待两个框架 stream 并尝试清空池内空闲空间。
 * 存活张量及已设置阈值继续保留。 */
DL_API dl_status dl_cuda_memory_release_unused(int32_t device);

DL_API dl_status dl_generator_create(uint64_t seed, dl_generator** output);
DL_API void dl_generator_release(dl_generator* generator);
DL_API dl_status dl_generator_seed(dl_generator* generator, uint64_t seed);
DL_API dl_status dl_manual_seed(uint64_t seed);
/* generator 为 NULL 时使用当前线程的默认随机流。 */
DL_API dl_status dl_uniform(dl_shape shape, double low, double high, dl_tensor_options options,
    dl_generator* generator, dl_tensor** output);
DL_API dl_status dl_randn(dl_shape shape, dl_tensor_options options, dl_generator* generator, dl_tensor** output);
DL_API dl_status dl_bernoulli(dl_shape shape, double probability, dl_device device, dl_generator* generator,
    dl_tensor** output);
/* 标签派生和分流是纯值操作；split 写入调用方提供的 count 个 key 槽。 */
DL_API dl_random_key dl_random_fold_in(dl_random_key key, uint64_t label);
DL_API void dl_random_split(dl_random_key key, size_t count, dl_random_key* outputs);
/* key 张量为 CPU 连续 int64[2]，两个整数保留 seed/stream 的原始位模式。 */
DL_API dl_status dl_random_key_tensor(dl_random_key key, dl_tensor** output);
DL_API dl_status dl_random_key_from_tensor(const dl_tensor* tensor, dl_random_key* output);
DL_API dl_status dl_uniform_with_key(dl_random_key key, dl_shape shape, double low, double high,
    dl_tensor_options options, dl_tensor** output);
DL_API dl_status dl_randn_with_key(dl_random_key key, dl_shape shape, dl_tensor_options options, dl_tensor** output);
DL_API dl_status dl_bernoulli_with_key(dl_random_key key, dl_shape shape, double probability,
    dl_device device, dl_tensor** output);

/* 模块与 Sequential 保留子模块的共享所有权；已加入的子模块句柄可由调用方释放。 */
DL_API void dl_module_release(dl_module* module);
DL_API dl_status dl_module_retain(const dl_module* module, dl_module** output);
DL_API dl_status dl_sequential_create(dl_module** output);
DL_API dl_status dl_sequential_append(dl_module* sequential, const char* name, const dl_module* child);
DL_API dl_status dl_linear_create(int64_t input_features, int64_t output_features, int32_t bias,
    dl_tensor_options options, dl_module** output);
DL_API dl_status dl_linear_from_tensors(const dl_tensor* weight, const dl_tensor* bias, dl_module** output);
DL_API dl_status dl_embedding_create(int64_t vocabulary, int64_t dimensions, dl_tensor_options options,
    dl_module** output);
DL_API dl_status dl_embedding_from_tensor(const dl_tensor* weight, dl_module** output);
DL_API dl_status dl_relu_create(dl_module** output);
DL_API dl_status dl_gelu_create(dl_module** output);
DL_API dl_status dl_sigmoid_create(dl_module** output);
DL_API dl_status dl_tanh_create(dl_module** output);
DL_API dl_status dl_silu_create(dl_module** output);
DL_API dl_status dl_softplus_create(dl_module** output);
DL_API dl_status dl_leaky_relu_create(double negative_slope, dl_module** output);
DL_API dl_status dl_softmax_create(int64_t axis, dl_module** output);
DL_API dl_status dl_log_softmax_create(int64_t axis, dl_module** output);
DL_API dl_status dl_flatten_create(int64_t first, int64_t last, dl_module** output);
DL_API dl_status dl_dropout_create(double probability, dl_module** output);
DL_API dl_status dl_layer_norm_create(dl_shape normalized_shape, double epsilon, int32_t affine,
    dl_tensor_options options, dl_module** output);
DL_API dl_status dl_group_norm_create(int64_t groups, int64_t channels, double epsilon, int32_t affine,
    dl_tensor_options options, dl_module** output);
DL_API dl_status dl_rms_norm_create(dl_shape normalized_shape, const double* epsilon, int32_t affine,
    dl_tensor_options options, dl_module** output);
DL_API dl_status dl_conv1d_create(int64_t input_channels, int64_t output_channels, int64_t kernel,
    dl_conv1d_options convolution, int32_t bias, dl_tensor_options options, dl_module** output);
DL_API dl_status dl_conv1d_from_tensors(const dl_tensor* weight, const dl_tensor* bias,
    dl_conv1d_options convolution, dl_module** output);
DL_API dl_status dl_max_pool1d_create(dl_pool1d_options options, dl_module** output);
DL_API dl_status dl_avg_pool1d_create(dl_pool1d_options options, int32_t count_include_pad, dl_module** output);
DL_API dl_status dl_adaptive_avg_pool1d_create(int64_t output_size, dl_module** output);
DL_API dl_status dl_adaptive_avg_pool2d_create(dl_size2 output_size, dl_module** output);
DL_API dl_status dl_upsample_create(dl_size2 output_size, dl_interpolation_mode mode, int32_t align_corners, dl_module** output);
DL_API dl_status dl_batch_norm1d_create(int64_t channels, dl_batch_norm_options normalization,
    dl_tensor_options options, dl_module** output);
DL_API dl_status dl_batch_norm1d_reset(dl_module* module);
DL_API dl_status dl_conv2d_create(int64_t input_channels, int64_t output_channels, dl_size2 kernel,
    dl_conv2d_options convolution, int32_t bias, dl_tensor_options options, dl_module** output);
DL_API dl_status dl_conv2d_from_tensors(const dl_tensor* weight, const dl_tensor* bias,
    dl_conv2d_options convolution, dl_module** output);
DL_API dl_status dl_max_pool2d_create(dl_pool2d_options options, dl_module** output);
DL_API dl_status dl_avg_pool2d_create(dl_pool2d_options options, int32_t count_include_pad, dl_module** output);
DL_API dl_status dl_batch_norm2d_create(int64_t channels, dl_batch_norm_options normalization,
    dl_tensor_options options, dl_module** output);
DL_API dl_status dl_batch_norm2d_reset(dl_module* module);
DL_API dl_status dl_attention_create(int64_t dimensions, int64_t heads, dl_attention_options attention,
    dl_tensor_options options, dl_module** output);
/* 在前向图之间配置对应类型的模块；融合因果计算通过标量选项表达，无需方形 mask。 */
DL_API dl_status dl_attention_set_backend(dl_module* module, dl_attention_backend backend, int32_t causal);
DL_API dl_status dl_transformer_layer_set_attention_backend(dl_module* module,
    dl_attention_backend backend, int32_t causal);
DL_API dl_status dl_transformer_set_attention_backend(dl_module* module,
    dl_attention_backend backend, int32_t causal);
DL_API dl_status dl_transformer_layer_create(int64_t dimensions, int64_t heads, int64_t feedforward,
    double dropout, dl_tensor_options options, dl_module** output);
DL_API dl_status dl_transformer_create(int64_t layers, int64_t dimensions, int64_t heads, int64_t feedforward,
    double dropout, dl_tensor_options options, dl_module** output);
DL_API dl_status dl_rnn_create(int64_t input_size, int64_t hidden_size, dl_recurrent_options recurrent,
    dl_rnn_activation activation, dl_tensor_options options, dl_module** output);
DL_API dl_status dl_gru_create(int64_t input_size, int64_t hidden_size, dl_recurrent_options recurrent,
    dl_tensor_options options, dl_module** output);
DL_API dl_status dl_lstm_create(int64_t input_size, int64_t hidden_size, dl_recurrent_options recurrent,
    dl_tensor_options options, dl_module** output);
/* 接受 RNN/GRU/LSTM 模块；初始 hidden/cell 为 NULL 时取零，RNN/GRU 的 cell 为 NULL。
 * 结果列表依次为 output、hidden、可选 cell；dl_module_forward 只返回序列 output。 */
DL_API dl_status dl_recurrent_forward(dl_module* module, const dl_tensor* input,
    const dl_tensor* hidden, const dl_tensor* cell, dl_tensor_list** output);
/* 函数式绑定保留原 Module 的共享所有权；结构登记在绑定存活期间保持稳定。
 * 初始状态是参数、buffer 和模式的独立快照，槽顺序由以下名称查询提供。 */
DL_API dl_status dl_functional_create(const dl_module* module, dl_functional** output);
DL_API void dl_functional_release(dl_functional* functional);
DL_API size_t dl_functional_parameter_count(const dl_functional* functional);
DL_API const char* dl_functional_parameter_name(const dl_functional* functional, size_t index);
DL_API size_t dl_functional_buffer_count(const dl_functional* functional);
DL_API const char* dl_functional_buffer_name(const dl_functional* functional, size_t index);
DL_API size_t dl_functional_module_count(const dl_functional* functional);
DL_API const char* dl_functional_module_name(const dl_functional* functional, size_t index);
DL_API dl_status dl_functional_state_create(const dl_functional* functional, dl_functional_state** output);
DL_API void dl_functional_state_release(dl_functional_state* state);
DL_API dl_status dl_functional_state_parameters(const dl_functional_state* state, dl_tensor_list** output);
DL_API dl_status dl_functional_state_buffers(const dl_functional_state* state, dl_tensor_list** output);
/* 设置槽复制 Tensor 句柄，保留传入身份；替换参数后由调用方安排与优化器的绑定。 */
DL_API dl_status dl_functional_state_set_parameter(dl_functional_state* state, size_t index, const dl_tensor* value);
DL_API dl_status dl_functional_state_set_buffer(dl_functional_state* state, size_t index, const dl_tensor* value);
DL_API int32_t dl_functional_state_training(const dl_functional_state* state, size_t module_index);
DL_API dl_status dl_functional_state_set_training(dl_functional_state* state, size_t module_index, int32_t enabled);
DL_API dl_status dl_functional_state_train(dl_functional_state* state, int32_t enabled);
/* 单输入前向交付拥有型输出和后继状态；后继状态共享参数身份，拥有独立的新 buffer。
 * 原状态保持原值，key 不推进 Generator；两个输出槽分别释放。 */
DL_API dl_status dl_functional_call(const dl_functional* functional, const dl_functional_state* state,
    const dl_tensor* input, dl_random_key key, dl_tensor** output, dl_functional_state** next_state);

DL_API dl_status dl_module_forward(dl_module* module, const dl_tensor* input, dl_tensor** output);
/* 专用 forward 接受对应构造器创建的模块，keep_mask 为 NULL 时保留全部位置。 */
DL_API dl_status dl_attention_forward(dl_module* module, const dl_tensor* query, const dl_tensor* key,
    const dl_tensor* value, const dl_tensor* keep_mask, dl_tensor** output);
DL_API dl_status dl_transformer_layer_forward(dl_module* module, const dl_tensor* input,
    const dl_tensor* keep_mask, dl_tensor** output);
DL_API dl_status dl_transformer_forward(dl_module* module, const dl_tensor* input,
    const dl_tensor* keep_mask, dl_tensor** output);
DL_API dl_status dl_module_train(dl_module* module, int32_t training);
DL_API int32_t dl_module_is_training(const dl_module* module);
/* 递归切换参数的求导状态，train/eval 独立管理，原有参数身份保持不变。 */
DL_API dl_status dl_module_set_requires_grad(dl_module* module, int32_t enabled);
DL_API dl_status dl_module_to(dl_module* module, dl_device device);

/* parameters 返回所有参数叶句柄，包含冻结项；trainable_parameters 只选择可求导项。
 * state 返回共享存储的参数和 buffer 数据，三个入口都保留原有名称顺序。 */
DL_API dl_status dl_module_parameters(const dl_module* module, dl_state** output);
DL_API dl_status dl_module_trainable_parameters(const dl_module* module, dl_state** output);
DL_API dl_status dl_module_state(const dl_module* module, dl_state** output);
/* 加载先保存与模型存储共享的输入值，再写回既有参数与 buffer。 */
DL_API dl_status dl_module_load_state(dl_module* module, const dl_state* state);
DL_API void dl_state_release(dl_state* state);
DL_API size_t dl_state_size(const dl_state* state);
/* 名称借用 state 的字符串；state_tensor 返回新的拥有型句柄。 */
DL_API const char* dl_state_name(const dl_state* state, size_t index);
DL_API dl_status dl_state_tensor(const dl_state* state, size_t index, dl_tensor** output);
DL_API dl_status dl_state_create(const char* const* names, const dl_tensor* const* tensors, size_t count,
    dl_state** output);
DL_API dl_status dl_state_save(const char* path, const dl_state* state);
DL_API dl_status dl_state_load(const char* path, dl_state** output);

/* 显式更新使用同 shape/dtype/device 的参数、梯度和缓冲，CPU f32/f64 或 CUDA f32。
 * 函数式入口返回独立连续参数与状态，新参数保留 requires_grad 且成为空梯度叶。
 * SGD 的 velocity 可为 NULL，结果为 [parameter, velocity]；无动量且无旧状态时只有 parameter。
 * Adam/AdamW 的 first/second 同时提供或同时为 NULL，step 是已完成的更新数，初次调用为零。
 * Adam/AdamW 结果为 [parameter, first, second, next_step]，next_step 是 CPU int64 标量。
 * 结果列表及取出的句柄分别拥有生命周期。显式梯度每次都执行一步，不读取参数梯度槽。 */
DL_API dl_status dl_sgd(const dl_tensor* parameter, const dl_tensor* gradient, const dl_tensor* velocity,
    dl_sgd_options options, dl_tensor_list** output);
DL_API dl_status dl_adamw(const dl_tensor* parameter, const dl_tensor* gradient,
    const dl_tensor* first, const dl_tensor* second, int64_t step,
    dl_adamw_options options, dl_tensor_list** output);
DL_API dl_status dl_adam(const dl_tensor* parameter, const dl_tensor* gradient,
    const dl_tensor* first, const dl_tensor* second, int64_t step,
    dl_adam_options options, dl_tensor_list** output);
/* RMSprop 的 square_average 为 NULL 时其他缓冲也为 NULL 且 step=0。
   已有状态可缺少 momentum/gradient_average，启用时创建，停用时保留传入值。
   输出为 [parameter, square_average, 存在的 momentum, 存在的 gradient_average, next_step]。
   某可选缓冲存在当且仅当此前已提供或本次启用，计数始终为最后一项。 */
DL_API dl_status dl_rmsprop(const dl_tensor* parameter, const dl_tensor* gradient,
    const dl_tensor* square_average, const dl_tensor* momentum, const dl_tensor* gradient_average,
    int64_t step, dl_rmsprop_options options, dl_tensor_list** output);
/* Adagrad 的 sum 可为 NULL，初始 step=0；输出为 [parameter, sum, next_step]。 */
DL_API dl_status dl_adagrad(const dl_tensor* parameter, const dl_tensor* gradient,
    const dl_tensor* sum, int64_t step, dl_adagrad_options options, dl_tensor_list** output);
/* 原地入口返回相同顺序的列表，参数和已有状态结果继续共享原身份；空状态在本次建立。
 * 参数为已完成相关反向图的可写叶，状态无求导历史；写入目标各自不重叠且与梯度分开。 */
DL_API dl_status dl_sgd_update(dl_tensor* parameter, const dl_tensor* gradient, dl_tensor* velocity,
    dl_sgd_options options, dl_tensor_list** output);
DL_API dl_status dl_adamw_update(dl_tensor* parameter, const dl_tensor* gradient,
    dl_tensor* first, dl_tensor* second, int64_t step,
    dl_adamw_options options, dl_tensor_list** output);
DL_API dl_status dl_adam_update(dl_tensor* parameter, const dl_tensor* gradient,
    dl_tensor* first, dl_tensor* second, int64_t step,
    dl_adam_options options, dl_tensor_list** output);
DL_API dl_status dl_rmsprop_update(dl_tensor* parameter, const dl_tensor* gradient,
    dl_tensor* square_average, dl_tensor* momentum, dl_tensor* gradient_average,
    int64_t step, dl_rmsprop_options options, dl_tensor_list** output);
DL_API dl_status dl_adagrad_update(dl_tensor* parameter, const dl_tensor* gradient,
    dl_tensor* sum, int64_t step, dl_adagrad_options options, dl_tensor_list** output);

DL_API dl_status dl_sgd_create(const dl_tensor* const* parameters, size_t count, dl_sgd_options options,
    dl_optimizer** output);
DL_API dl_status dl_adamw_create(const dl_tensor* const* parameters, size_t count, dl_adamw_options options,
    dl_optimizer** output);
DL_API dl_status dl_adam_create(const dl_tensor* const* parameters, size_t count, dl_adam_options options,
    dl_optimizer** output);
DL_API dl_status dl_rmsprop_create(const dl_tensor* const* parameters, size_t count, dl_rmsprop_options options,
    dl_optimizer** output);
DL_API dl_status dl_adagrad_create(const dl_tensor* const* parameters, size_t count, dl_adagrad_options options,
    dl_optimizer** output);
DL_API dl_status dl_sgd_create_groups(const dl_parameter_group* groups, size_t count, dl_sgd_options options,
    dl_optimizer** output);
DL_API dl_status dl_adamw_create_groups(const dl_parameter_group* groups, size_t count, dl_adamw_options options,
    dl_optimizer** output);
DL_API dl_status dl_adam_create_groups(const dl_parameter_group* groups, size_t count, dl_adam_options options,
    dl_optimizer** output);
DL_API dl_status dl_rmsprop_create_groups(const dl_parameter_group* groups, size_t count, dl_rmsprop_options options,
    dl_optimizer** output);
DL_API dl_status dl_adagrad_create_groups(const dl_parameter_group* groups, size_t count, dl_adagrad_options options,
    dl_optimizer** output);
DL_API void dl_optimizer_release(dl_optimizer* optimizer);
DL_API dl_status dl_optimizer_zero_grad(dl_optimizer* optimizer);
DL_API dl_status dl_optimizer_step(dl_optimizer* optimizer);
DL_API size_t dl_optimizer_group_count(const dl_optimizer* optimizer);
DL_API double dl_optimizer_learning_rate(const dl_optimizer* optimizer, size_t group);
DL_API double dl_optimizer_weight_decay(const dl_optimizer* optimizer, size_t group);
DL_API dl_status dl_optimizer_set_learning_rate(dl_optimizer* optimizer, size_t group, double value);
DL_API dl_status dl_optimizer_set_weight_decay(dl_optimizer* optimizer, size_t group, double value);
DL_API dl_status dl_clip_grad_norm(const dl_tensor* const* parameters, size_t count, double max_norm,
    double order, dl_tensor** total_norm);
DL_API dl_status dl_clip_grad_value(const dl_tensor* const* parameters, size_t count, double value);
/* 直接处理梯度值，保留空项及重复身份，只对每个身份计数和计算一次。
   纯入口返回独立连续结果，原地入口使用可写且彼此不重叠的各身份存储。
   CPU 为 f32/f64，CUDA 为 f32；界限非负，order 为正数或正无穷。
   范数入口的非空项同 dtype、设备；全空列表的 total_norm 为 CPU f32 标量零。 */
DL_API dl_status dl_clip_gradients_norm(const dl_gradients* gradients, double max_norm, double order,
    dl_gradients** output, dl_tensor** total_norm);
DL_API dl_status dl_clip_gradients_norm_update(dl_gradients* gradients, double max_norm, double order,
    dl_tensor** total_norm);
DL_API dl_status dl_clip_gradients_value(const dl_gradients* gradients, double value, dl_gradients** output);
DL_API dl_status dl_clip_gradients_value_update(dl_gradients* gradients, double value);

/* 调度器保留优化器的共享所有权；在 optimizer_step 后调用 scheduler_step。 */
DL_API dl_status dl_step_lr_create(dl_optimizer* optimizer, int64_t interval, double gamma, dl_scheduler** output);
DL_API dl_status dl_multistep_lr_create(dl_optimizer* optimizer, dl_shape milestones, double gamma, dl_scheduler** output);
DL_API dl_status dl_exponential_lr_create(dl_optimizer* optimizer, double gamma, dl_scheduler** output);
DL_API dl_status dl_cosine_lr_create(dl_optimizer* optimizer, int64_t period, double minimum_rate, dl_scheduler** output);
DL_API dl_status dl_linear_lr_create(dl_optimizer* optimizer, int64_t steps, double start_factor,
    double end_factor, dl_scheduler** output);
DL_API void dl_scheduler_release(dl_scheduler* scheduler);
DL_API dl_status dl_scheduler_step(dl_scheduler* scheduler);
DL_API dl_status dl_scheduler_set_step(dl_scheduler* scheduler, int64_t completed_steps);
DL_API int64_t dl_scheduler_get_step(const dl_scheduler* scheduler);

/* loader 保留数据和显式 generator 的所有权；默认随机流随当前线程存活。 */
DL_API dl_status dl_loader_create(const dl_tensor* input, const dl_tensor* target, dl_loader_options options,
    dl_generator* generator, dl_loader** output);
DL_API void dl_loader_release(dl_loader* loader);
DL_API dl_status dl_loader_reset(dl_loader* loader);
/* 轮次结束返回两个 NULL；正常批次返回两个独立拥有的张量句柄。 */
DL_API dl_status dl_loader_next(dl_loader* loader, dl_tensor** input, dl_tensor** target);

/* 传输持有源与目标；tensor 将 CUDA 结果接入计算流，CPU 结果等待复制完成。 */
DL_API dl_status dl_pin_memory(const dl_tensor* input, dl_tensor** output);
DL_API int dl_tensor_is_pinned(const dl_tensor* input);
DL_API dl_status dl_transfer_create(const dl_tensor* input, dl_device target, dl_transfer** output);
DL_API void dl_transfer_release(dl_transfer* transfer);
DL_API dl_status dl_transfer_tensor(const dl_transfer* transfer, dl_tensor** output);
DL_API dl_status dl_transfer_ready(const dl_transfer* transfer, int* ready);
DL_API dl_status dl_transfer_wait(const dl_transfer* transfer);

/* 预取器保留源 loader 和生成器的所有权，depth 为正；迭代与重置通过预取器执行。 */
DL_API dl_status dl_prefetch_loader_create(dl_loader* source, dl_device target, size_t depth, dl_prefetch_loader** output);
DL_API void dl_prefetch_loader_release(dl_prefetch_loader* loader);
DL_API dl_status dl_prefetch_loader_next(dl_prefetch_loader* loader, dl_tensor** input, dl_tensor** target);
DL_API dl_status dl_prefetch_loader_reset(dl_prefetch_loader* loader);

DL_API void dl_checkpoint_release(dl_checkpoint* checkpoint);
/* 优化器与 functional state 的完整参数槽按身份一一对应，登记顺序可以不同。
 * 采集按身份同时重排状态槽和组索引，包含冻结及尚无状态的参数。
 * 恢复交付新状态和绑定这些新叶的优化器，原 Module 及原参数、buffer 保持原值。 */
DL_API dl_status dl_checkpoint_capture_functional(const dl_functional* functional,
    const dl_functional_state* state, const dl_optimizer* optimizer, dl_random_key key,
    dl_training_progress progress, const dl_loader* loader, const dl_named_generator* generators,
    size_t count, dl_checkpoint** output);
DL_API dl_status dl_checkpoint_restore_functional(const dl_checkpoint* checkpoint, const dl_functional* functional,
    dl_device device, dl_loader* loader, const dl_named_generator* generators, size_t count,
    dl_functional_state** state, dl_optimizer** optimizer, dl_random_key* key, dl_training_progress* progress);
DL_API dl_status dl_checkpoint_capture_functional_prefetch(const dl_functional* functional,
    const dl_functional_state* state, const dl_optimizer* optimizer, dl_random_key key,
    dl_training_progress progress, const dl_prefetch_loader* loader, const dl_named_generator* generators,
    size_t count, dl_checkpoint** output);
DL_API dl_status dl_checkpoint_restore_functional_prefetch(const dl_checkpoint* checkpoint, const dl_functional* functional,
    dl_device device, dl_prefetch_loader* loader, const dl_named_generator* generators, size_t count,
    dl_functional_state** state, dl_optimizer** optimizer, dl_random_key* key, dl_training_progress* progress);
/* v6 保存可选 key；旧文件或未附加 key 的普通快照使 present 为零，output 保持原值。 */
DL_API dl_status dl_checkpoint_set_random_key(dl_checkpoint* checkpoint, dl_random_key key);
DL_API dl_status dl_checkpoint_random_key(const dl_checkpoint* checkpoint, dl_random_key* output, int32_t* present);
DL_API dl_status dl_checkpoint_capture(const dl_module* model, const dl_optimizer* optimizer,
    dl_training_progress progress, const dl_loader* loader, const dl_named_generator* generators, size_t count,
    dl_checkpoint** output);
DL_API dl_status dl_checkpoint_restore(const dl_checkpoint* checkpoint, dl_module* model, dl_optimizer* optimizer,
    dl_loader* loader, const dl_named_generator* generators, size_t count, dl_training_progress* progress);
DL_API dl_status dl_checkpoint_capture_prefetch(const dl_module* model, const dl_optimizer* optimizer,
    dl_training_progress progress, const dl_prefetch_loader* loader, const dl_named_generator* generators,
    size_t count, dl_checkpoint** output);
DL_API dl_status dl_checkpoint_restore_prefetch(const dl_checkpoint* checkpoint, dl_module* model,
    dl_optimizer* optimizer, dl_prefetch_loader* loader, const dl_named_generator* generators,
    size_t count, dl_training_progress* progress);
DL_API dl_status dl_checkpoint_save(const char* path, const dl_checkpoint* checkpoint);
DL_API dl_status dl_checkpoint_load(const char* path, dl_checkpoint** output);
DL_API dl_status dl_checkpoint_extra(const dl_checkpoint* checkpoint, dl_state** output);
DL_API dl_status dl_checkpoint_set_extra(dl_checkpoint* checkpoint, const dl_state* state);
/* 保存前附加缩放状态，恢复后读入；旧快照无此字段时 restored 为零，scaler 保持原配置。 */
DL_API dl_status dl_checkpoint_set_grad_scaler(dl_checkpoint* checkpoint, const dl_grad_scaler* scaler);
DL_API dl_status dl_checkpoint_restore_grad_scaler(const dl_checkpoint* checkpoint, dl_grad_scaler* scaler, int32_t* restored);
/* 保存前按名称登记调度器；先恢复模型和优化器，再恢复相应调度器。 */
DL_API dl_status dl_checkpoint_set_scheduler(dl_checkpoint* checkpoint, const char* name, const dl_scheduler* scheduler);
DL_API dl_status dl_checkpoint_restore_scheduler(const dl_checkpoint* checkpoint, const char* name, dl_scheduler* scheduler);

#ifdef __cplusplus
}
#endif
#endif

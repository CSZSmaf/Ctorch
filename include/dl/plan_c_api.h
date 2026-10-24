#ifndef CTORCH_PLAN_C_API_H
#define CTORCH_PLAN_C_API_H

#include "dl/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dl_graph dl_graph;
typedef struct dl_plan_operation dl_plan_operation;
typedef struct dl_plan_values dl_plan_values;
typedef struct dl_execution_plan dl_execution_plan;
typedef struct dl_execution_workspace dl_execution_workspace;
typedef struct dl_plan_infer_context dl_plan_infer_context;

/* 值只在创建它的图中使用；复制或取出值不持有图，也不分配张量。 */
typedef struct dl_plan_value { size_t index; } dl_plan_value;
typedef enum dl_plan_effect {
    DL_PLAN_PURE = 0, DL_PLAN_RANDOM = 1, DL_PLAN_MUTATION = 2, DL_PLAN_EXTERNAL = 3
} dl_plan_effect;
typedef enum dl_plan_builtin {
    DL_PLAN_ADD = 0, DL_PLAN_ADD_RELU, DL_PLAN_MUL, DL_PLAN_MATMUL,
    DL_PLAN_RELU, DL_PLAN_SILU, DL_PLAN_CONTIGUOUS,
    /* LINEAR 接收 input、weight 和可选 bias，权重排列为 [out, in]。 */
    DL_PLAN_LINEAR
} dl_plan_builtin;

/* 两个别名字段取 -1 表示独立存储；至多一个字段为非负的输入或更早输出序号。 */
typedef struct dl_plan_output_spec {
    dl_tensor_spec tensor;
    int64_t alias_input, alias_output;
} dl_plan_output_spec;
typedef struct dl_plan_compile_options { int32_t fuse_pointwise; } dl_plan_compile_options;
typedef struct dl_plan_info {
    size_t input_count, output_count, step_count, slot_count, fusion_count;
    size_t storage_count, buffer_count;
} dl_plan_info;
typedef struct dl_plan_storage_lifetime {
    dl_plan_value origin;
    size_t first_step, last_step;
    int32_t external;
} dl_plan_storage_lifetime;
typedef struct dl_plan_buffer_requirement { dl_device device; size_t bytes, count; } dl_plan_buffer_requirement;
typedef struct dl_workspace_memory_stats {
    size_t buffers, in_use_buffers, reserved_bytes, in_use_bytes;
} dl_workspace_memory_stats;

/* infer 只读取规格，按输出次序向 context 追加描述，追加时复制 shape/strides。
 * execute 借用输入句柄，将拥有型张量列表交给框架；成功时列表数量及规格与 infer 一致。
 * 零输出也交付空列表；失败时已交付的列表由框架释放，错误文本规则与 custom VJP 相同。 */
typedef dl_status (*dl_plan_infer_fn)(dl_plan_infer_context* context, const dl_tensor_spec* inputs,
    size_t count, void* user_data, const char** error_message);
typedef dl_status (*dl_plan_execute_fn)(const dl_tensor* const* inputs, size_t count,
    void* user_data, dl_tensor_list** output, const char** error_message);
/* 工作区在当前线程同步调用此回调；user_data 只借用到回调结束，结果可保存在其中。
 * 返回 DL_OK 表示完成；失败文本可来自静态数据、user_data 或 dl_last_error()。 */
typedef dl_status (*dl_workspace_fn)(void* user_data, const char** error_message);

/* 每个操作可重复用于多张图，graph_call 复制操作描述。
 * 无参数工厂与下面的选项工厂直接使用对应 C++ plan::ops。 */
DL_API dl_status dl_plan_builtin_operation(dl_plan_builtin kind, dl_plan_operation** output);
/* 卷积描述固定 stride、padding、dilation 与 groups，输入按 input、weight、可选 bias 排列。
 * 结果按需整理为规范连续布局，规格保持独立于后端和 autocast 的物理布局选择。 */
DL_API dl_status dl_plan_conv1d(dl_conv1d_options options, dl_plan_operation** output);
DL_API dl_status dl_plan_conv2d(dl_conv2d_options options, dl_plan_operation** output);
/* 池化接收一个输入，输出为独立连续结果；计数选项和输出尺寸在建图时确定。 */
DL_API dl_status dl_plan_max_pool1d(dl_pool1d_options options, dl_plan_operation** output);
DL_API dl_status dl_plan_avg_pool1d(dl_pool1d_options options, int32_t count_include_pad,
    dl_plan_operation** output);
DL_API dl_status dl_plan_max_pool2d(dl_pool2d_options options, dl_plan_operation** output);
DL_API dl_status dl_plan_avg_pool2d(dl_pool2d_options options, int32_t count_include_pad,
    dl_plan_operation** output);
DL_API dl_status dl_plan_adaptive_avg_pool1d(int64_t output_size, dl_plan_operation** output);
DL_API dl_status dl_plan_adaptive_avg_pool2d(dl_size2 output_size, dl_plan_operation** output);
/* 图像插值接收一个 NCHW/CHW 输入，模式与对齐规则沿用 dl_interpolate。 */
DL_API dl_status dl_plan_interpolate(dl_size2 output_size, dl_interpolation_mode mode, int32_t align_corners,
    dl_plan_operation** output);
DL_API dl_status dl_plan_reshape(dl_shape shape, dl_plan_operation** output);
DL_API dl_status dl_plan_transpose(int64_t first, int64_t second, dl_plan_operation** output);
DL_API dl_status dl_plan_dropout(double probability, int32_t training, dl_plan_operation** output);
DL_API dl_status dl_plan_batch_norm(dl_batch_norm_options options, int32_t training, dl_plan_operation** output);
DL_API dl_status dl_plan_functional_batch_norm(dl_batch_norm_options options, int32_t training,
    dl_plan_operation** output);
/* 包装已绑定的单输入模块；图输入为 input、parameters、buffers、CPU int64[2] key。
 * training 按模块槽复制，count 与 module_count 一致。infer 读取完整规格并只追加模型的一个输出，
 * buffer 规格表示本次连续副本，指向 buffer 输入的别名自动关联对应新 buffer。
 * 图结果依次为 next_buffers、模型输出；操作持有模块所有权，可随后释放 functional 句柄。
 * 创建调用接管 user_data，deleter 在最后一个相关描述、图或计划结束时执行。 */
DL_API dl_status dl_plan_functional_module(const dl_functional* functional, const uint8_t* training, size_t count,
    dl_plan_infer_fn infer, void* user_data, dl_user_data_deleter deleter, dl_plan_operation** output);
DL_API dl_status dl_plan_unbind(int64_t axis, dl_plan_operation** output);
DL_API dl_status dl_plan_lstm_cell(int32_t bias, dl_plan_operation** output);
DL_API dl_status dl_plan_fold_in_key(uint64_t label, dl_plan_operation** output);
DL_API dl_status dl_plan_split_key(size_t count, dl_plan_operation** output);
DL_API dl_status dl_plan_uniform(dl_shape shape, double low, double high, dl_tensor_options options,
    dl_plan_operation** output);
DL_API dl_status dl_plan_randn(dl_shape shape, dl_tensor_options options, dl_plan_operation** output);
DL_API dl_status dl_plan_bernoulli(dl_shape shape, double probability, dl_device device, dl_plan_operation** output);
DL_API dl_status dl_plan_dropout_with_key(double probability, int32_t training, dl_plan_operation** output);
/* 输入为 Q/K/V、可选 bool mask、可选 CPU key，顺序与 C++ ops::attention 一致。 */
DL_API dl_status dl_plan_attention(dl_attention_compute_options options, int32_t masked, int32_t keyed,
    dl_plan_operation** output);
/* 显式状态和各结果顺序沿用函数式优化器，已有状态的步数输入为 CPU int64 标量。 */
DL_API dl_status dl_plan_sgd(dl_sgd_options options, dl_plan_operation** output);
DL_API dl_status dl_plan_adamw(dl_adamw_options options, dl_plan_operation** output);
DL_API dl_status dl_plan_adam(dl_adam_options options, dl_plan_operation** output);
DL_API dl_status dl_plan_rmsprop(dl_rmsprop_options options, dl_plan_operation** output);
DL_API dl_status dl_plan_adagrad(dl_adagrad_options options, dl_plan_operation** output);

/* 创建调用接管 user_data；可选 deleter 在最后一个相关操作、图、计划或工作区结束时执行。
 * 自定义描述使用 custom 语义；effect 声明随机流、输入写入或外部影响。
 * execute 使用普通 C 算子建立求导关系，也可在其中调用 dl_custom_vjp。 */
DL_API dl_status dl_plan_custom_operation(const char* name, dl_plan_infer_fn infer,
    dl_plan_execute_fn execute, dl_plan_effect effect, void* user_data,
    dl_user_data_deleter deleter, dl_plan_operation** output);
DL_API dl_status dl_plan_infer_add_output(dl_plan_infer_context* context, dl_plan_output_spec spec);
DL_API dl_status dl_plan_infer_contiguous_output(dl_plan_infer_context* context,
    dl_shape shape, dl_dtype dtype, dl_device device);
DL_API void dl_plan_operation_release(dl_plan_operation* operation);

DL_API dl_status dl_graph_create(dl_graph** output);
DL_API void dl_graph_release(dl_graph* graph);
/* 输入元数据及名称在调用时复制；name 为 NULL 时使用空名称。 */
DL_API dl_status dl_graph_input(dl_graph* graph, dl_tensor_spec spec, const char* name, dl_plan_value* output);
DL_API dl_status dl_graph_input_contiguous(dl_graph* graph, dl_shape shape, dl_dtype dtype,
    dl_device device, const char* name, dl_plan_value* output);
/* 规格借用图的元数据，有效到下次修改该图或释放图。 */
DL_API dl_tensor_spec dl_graph_get_spec(const dl_graph* graph, dl_plan_value value);
DL_API dl_status dl_graph_call(dl_graph* graph, const dl_plan_operation* operation,
    const dl_plan_value* inputs, size_t count, dl_plan_values** output);
DL_API void dl_plan_values_release(dl_plan_values* values);
DL_API size_t dl_plan_values_size(const dl_plan_values* values);
DL_API dl_plan_value dl_plan_values_get(const dl_plan_values* values, size_t index);

DL_API dl_plan_compile_options dl_default_plan_compile_options(void);
/* 编译只处理元数据；options 为 NULL 时使用默认值，图及操作句柄之后可以释放。 */
DL_API dl_status dl_graph_compile(const dl_graph* graph, const dl_plan_value* outputs, size_t count,
    const dl_plan_compile_options* options, dl_execution_plan** output);
DL_API void dl_execution_plan_release(dl_execution_plan* plan);
DL_API dl_status dl_execution_plan_retain(const dl_execution_plan* plan, dl_execution_plan** output);
DL_API dl_plan_info dl_execution_plan_get_info(const dl_execution_plan* plan);
/* 计划规格借用不可变元数据，有效到该计划句柄释放。 */
DL_API dl_tensor_spec dl_execution_plan_input_spec(const dl_execution_plan* plan, size_t index);
DL_API dl_tensor_spec dl_execution_plan_output_spec(const dl_execution_plan* plan, size_t index);
DL_API dl_plan_storage_lifetime dl_execution_plan_storage(const dl_execution_plan* plan, size_t index);
DL_API dl_plan_buffer_requirement dl_execution_plan_buffer(const dl_execution_plan* plan, size_t index);
DL_API dl_status dl_execution_plan_run(const dl_execution_plan* plan,
    const dl_tensor* const* inputs, size_t count, dl_tensor_list** output);
/* 单个浮点输出的只读计划可作为激活重算块；随机流约定沿用 dl_activation_checkpoint。
 * 所有求导参数通过 inputs 绑定，持久状态更新放在计划外。
 * 返回普通拥有型张量，重算节点自行持有计划，原计划句柄之后可以释放。 */
DL_API dl_status dl_execution_plan_checkpoint(const dl_execution_plan* plan,
    const dl_tensor* const* inputs, size_t count, dl_checkpoint_options options, dl_tensor** output);

/* prepare 实际预留 CPU/CUDA 存储；工作区持有计划，可在创建后释放计划句柄。
 * 同一工作区串行使用，run 不递归进入自身；返回张量按普通所有权存活。 */
DL_API dl_status dl_execution_plan_prepare(const dl_execution_plan* plan, dl_execution_workspace** output);
DL_API void dl_execution_workspace_release(dl_execution_workspace* workspace);
DL_API dl_status dl_execution_workspace_run(dl_execution_workspace* workspace,
    const dl_tensor* const* inputs, size_t count, dl_tensor_list** output);
/* function 内的普通张量分配共用工作区，可包含 run、损失、反向和优化器更新。
 * 分配作用域可嵌套，结束时恢复外层池；工作区在回调期间保持存活。
 * 此调用只选择缓冲池，梯度模式、精度及 CUDA stream 顺序沿用现有接口。 */
DL_API dl_status dl_execution_workspace_with_memory(dl_execution_workspace* workspace,
    dl_workspace_fn function, void* user_data);
DL_API dl_workspace_memory_stats dl_execution_workspace_memory_stats(
    const dl_execution_workspace* workspace, dl_device device);
/* 只归还本工作区的空闲块，CUDA 释放沿原计算 stream 提交。 */
DL_API dl_status dl_execution_workspace_release_unused_memory(dl_execution_workspace* workspace);

#ifdef __cplusplus
}
#endif

#endif

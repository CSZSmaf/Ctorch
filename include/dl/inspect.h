#pragma once

#include "dl/nn.h"
#include "dl/plan.h"

#include <string>

namespace dl {

/// 张量的形状、布局、精度、设备与求导属性；只读取元信息，不同步设备或复制数值。
DL_API std::string describe(const Tensor& tensor);
/// 模块路径、唯一参数量、冻结状态和共享名称，不读取参数数值。
DL_API std::string summary(const nn::Module& model);
/// 已编译计划的操作数、融合数及前向缓冲需求，不创建执行工作区。
DL_API std::string describe(const plan::ExecutionPlan& plan);

} // namespace dl

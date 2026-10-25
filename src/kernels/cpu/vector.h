#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 连续向量的缩放累加：y[i] += alpha * x[i]。 */
void dl_axpy_f32(size_t n, float alpha, const float* x, float* y);
void dl_axpy_f64(size_t n, double alpha, const double* x, double* y);

/* AdamW 在同一遍历中更新一阶矩、二阶矩和参数，偏差校正由调用方给定。 */
void dl_adamw_f32(size_t n, float* parameter, const float* gradient, float* first, float* second,
                  float rate, double beta1, double beta2, float epsilon, float decay,
                  float inverse_bias1, float inverse_bias2);
void dl_adamw_f64(size_t n, double* parameter, const double* gradient, double* first, double* second,
                  double rate, double beta1, double beta2, double epsilon, double decay,
                  double inverse_bias1, double inverse_bias2);

#ifdef __cplusplus
}
#endif

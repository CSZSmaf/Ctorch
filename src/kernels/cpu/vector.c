#include "vector.h"

#include <math.h>

void dl_axpy_f32(size_t n, float alpha, const float* x, float* y) {
    for (size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

void dl_axpy_f64(size_t n, double alpha, const double* x, double* y) {
    for (size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

void dl_adamw_f32(size_t n, float* parameter, const float* gradient, float* first, float* second,
                  float rate, double beta1, double beta2, float epsilon, float decay,
                  float inverse_bias1, float inverse_bias2) {
    const float b1 = (float)beta1, b2 = (float)beta2;
    const float w1 = (float)(1.0 - beta1), w2 = (float)(1.0 - beta2);
    for (size_t i = 0; i < n; ++i) {
        const float g = gradient[i];
        first[i] = b1 * first[i] + w1 * g;
        second[i] = b2 * second[i] + w2 * g * g;
        const float direction = first[i] * inverse_bias1
            / (sqrtf(second[i] * inverse_bias2) + epsilon);
        parameter[i] = (1.0f - rate * decay) * parameter[i] - rate * direction;
    }
}

void dl_adamw_f64(size_t n, double* parameter, const double* gradient, double* first, double* second,
                  double rate, double beta1, double beta2, double epsilon, double decay,
                  double inverse_bias1, double inverse_bias2) {
    for (size_t i = 0; i < n; ++i) {
        const double g = gradient[i];
        first[i] = beta1 * first[i] + (1.0 - beta1) * g;
        second[i] = beta2 * second[i] + (1.0 - beta2) * g * g;
        const double direction = first[i] * inverse_bias1
            / (sqrt(second[i] * inverse_bias2) + epsilon);
        parameter[i] = (1.0 - rate * decay) * parameter[i] - rate * direction;
    }
}

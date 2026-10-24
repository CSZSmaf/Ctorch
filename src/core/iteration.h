#pragma once

#include "core/tensor_data.h"

#include <array>
#include <utility>

namespace dl::detail {

/// 一个逻辑迭代空间对应多个独立步长。零步长表示广播或归约目标的复用。
template <std::size_t Operands> class IterationPlan {
public:
    IterationPlan(Shape shape, std::array<Shape, Operands> operand_strides)
        : shape_(std::move(shape)), strides_(std::move(operand_strides)), count_(numel(shape_)) {
        for (std::size_t dim = shape_.size(); dim-- > 0;) {
            if (shape_[dim] == 1) {
                erase_dimension(dim);
            }
        }
        for (Index dim = static_cast<Index>(shape_.size()) - 2; dim >= 0; --dim) {
            // 所有操作数都连续衔接的维度可以共同合并，逻辑访问顺序保持不变。
            bool adjacent = true;
            for (const auto& strides : strides_) {
                adjacent &= strides[dim] == strides[dim + 1] * shape_[dim + 1];
            }
            if (adjacent) {
                shape_[dim] *= shape_[dim + 1];
                for (auto& strides : strides_) {
                    strides[dim] = strides[dim + 1];
                }
                erase_dimension(static_cast<std::size_t>(dim + 1));
            }
        }
    }

    /// 内层共享固定步长，外层负责坐标进位；标量张量作为长度一的行。
    template <class Function> void for_each_row(Function&& function) const {
        if (count_ == 0) return;
        std::array<Index, Operands> offsets{}, steps{};
        for (std::size_t operand = 0; operand < Operands; ++operand)
            steps[operand] = shape_.empty() ? 0 : strides_[operand].back();
        if (shape_.size() <= 1) {
            function(0, count_, offsets, steps);
            return;
        }

        const auto width = shape_.back();
        Shape coordinates(shape_.size() - 1, 0);
        for (Index index = 0; index < count_; index += width) {
            function(index, width, offsets, steps);
            for (std::size_t dim = coordinates.size(); dim-- > 0;) {
                ++coordinates[dim];
                for (std::size_t operand = 0; operand < Operands; ++operand) {
                    offsets[operand] += strides_[operand][dim];
                }
                if (coordinates[dim] < shape_[dim]) {
                    break;
                }
                coordinates[dim] = 0;
                for (std::size_t operand = 0; operand < Operands; ++operand) {
                    offsets[operand] -= shape_[dim] * strides_[operand][dim];
                }
            }
        }
    }

    template <class Function> void for_each(Function&& function) const {
        for_each_row([&](Index base, Index width, auto offsets, const auto& steps) {
            for (Index column = 0; column < width; ++column) {
                function(base + column, offsets);
                for (std::size_t operand = 0; operand < Operands; ++operand)
                    offsets[operand] += steps[operand];
            }
        });
    }

private:
    void erase_dimension(std::size_t dim) {
        shape_.erase(shape_.begin() + static_cast<Index>(dim));
        for (auto& strides : strides_) {
            strides.erase(strides.begin() + static_cast<Index>(dim));
        }
    }

    Shape shape_;
    std::array<Shape, Operands> strides_;
    Index count_;
};

} // namespace dl::detail

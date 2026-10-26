#include "dl/decoder.h"
#include "core/tensor_impl.h"
#include "ops/numeric.h"

#include <stdexcept>

namespace dl::nn {

KVCache::KVCache(Index capacity) : capacity_(capacity) {
    if (capacity <= 0) throw std::invalid_argument("KV cache capacity must be positive");
}
void KVCache::append(const Tensor& key, const Tensor& value) {
    const auto count = key.size(-2);
    if (count != value.size(-2) || count > capacity_ - length_)
        throw std::invalid_argument("KV cache append exceeds capacity or has unequal key/value lengths");
    auto reserve = [&](const Tensor& input, std::optional<Tensor>& storage) {
        auto shape = input.shape();
        shape[shape.size() - 2] = capacity_;
        if (!storage) storage = detail::make_tensor(detail::empty_data(shape, input.dtype(), input.device()));
        else if (storage->shape() != shape || storage->dtype() != input.dtype() || storage->device() != input.device())
            throw std::invalid_argument("KV cache shape, dtype or device changed");
    };
    reserve(key, key_); reserve(value, value_);
    auto copy = [&](const Tensor& input, Tensor& storage) {
        auto target = detail::numeric::slice(detail::data_of(storage), input.ndim() - 2, length_, length_ + count, 1);
        detail::numeric::copy_into(target, detail::data_of(input));
    };
    copy(key, *key_); copy(value, *value_);
    length_ += count;
}
Tensor KVCache::key() const {
    if (!key_) throw std::logic_error("KV cache has no key tensor before its first append");
    return key_->narrow(-2, 0, length_);
}
Tensor KVCache::value() const {
    if (!value_) throw std::logic_error("KV cache has no value tensor before its first append");
    return value_->narrow(-2, 0, length_);
}
void KVCache::truncate(Index length) {
    if (length < 0 || length > length_) throw std::out_of_range("KV cache truncation must keep an existing prefix");
    length_ = length;
}

} // namespace dl::nn

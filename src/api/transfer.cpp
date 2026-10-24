#include "api/common.h"

using namespace dl::capi;

extern "C" {

dl_status dl_pin_memory(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.pin_memory(); });
}
int dl_tensor_is_pinned(const dl_tensor* input) { return input->value.is_pinned(); }
dl_status dl_transfer_create(const dl_tensor* input, dl_device target, dl_transfer** output) {
    return make_handle(output, [&] { return dl::Transfer(input->value, convert(target)); });
}
void dl_transfer_release(dl_transfer* transfer) { delete transfer; }
dl_status dl_transfer_tensor(const dl_transfer* transfer, dl_tensor** output) {
    return make_handle(output, [&] { return transfer->value.tensor(); });
}
dl_status dl_transfer_ready(const dl_transfer* transfer, int* ready) {
    return invoke([&] { *ready = transfer->value.ready(); });
}
dl_status dl_transfer_wait(const dl_transfer* transfer) { return invoke([&] { transfer->value.wait(); }); }

dl_status dl_prefetch_loader_create(dl_loader* source, dl_device target, size_t depth, dl_prefetch_loader** output) {
    *output = nullptr;
    return invoke([&] {
        auto value = std::make_unique<dl::data::PrefetchLoader>(*source->value, convert(target), depth);
        *output = new dl_prefetch_loader{source->generator, source->value, std::move(value)};
    });
}
void dl_prefetch_loader_release(dl_prefetch_loader* loader) { delete loader; }
dl_status dl_prefetch_loader_reset(dl_prefetch_loader* loader) { return invoke([&] { loader->value->reset(); }); }
dl_status dl_prefetch_loader_next(dl_prefetch_loader* loader, dl_tensor** input, dl_tensor** target) {
    *input = nullptr;
    *target = nullptr;
    return invoke([&] {
        if (auto batch = loader->value->next()) {
            auto x = std::make_unique<dl_tensor>(dl_tensor{std::move(batch->input)});
            auto y = std::make_unique<dl_tensor>(dl_tensor{std::move(batch->target)});
            *input = x.release();
            *target = y.release();
        }
    });
}

} // extern "C"

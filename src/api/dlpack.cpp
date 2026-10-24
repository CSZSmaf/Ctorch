#include "dl/dlpack_c_api.h"
#include "dl/dlpack.h"
#include "api/common.h"

extern "C" {

dl_status dl_dlpack_consumer_stream(dl_device device, intptr_t* stream) {
    return dl::capi::invoke([&] { *stream = dl::dlpack_consumer_stream(dl::capi::convert(device)); });
}

dl_status dl_to_dlpack(const dl_tensor* tensor, intptr_t consumer_stream, DLManagedTensorVersioned** output) {
    *output = nullptr;
    return dl::capi::invoke([&] { *output = dl::to_dlpack(tensor->value, consumer_stream).release(); });
}

dl_status dl_to_dlpack_legacy(const dl_tensor* tensor, intptr_t consumer_stream, DLManagedTensor** output) {
    *output = nullptr;
    return dl::capi::invoke([&] { *output = dl::to_dlpack_legacy(tensor->value, consumer_stream).release(); });
}

dl_status dl_from_dlpack(DLManagedTensorVersioned* tensor, dl_tensor** output) {
    *output = nullptr;
    // 拥有权先于输出句柄分配建立，分配或导入失败同样只释放一次生产者描述符。
    dl::DLPackTensor owner(tensor);
    return dl::capi::invoke([&] {
        auto value = dl::from_dlpack(std::move(owner));
        *output = new dl_tensor{std::move(value)};
    });
}

dl_status dl_from_dlpack_legacy(DLManagedTensor* tensor, dl_tensor** output) {
    *output = nullptr;
    dl::LegacyDLPackTensor owner(tensor);
    return dl::capi::invoke([&] {
        auto value = dl::from_dlpack(std::move(owner));
        *output = new dl_tensor{std::move(value)};
    });
}

void dl_dlpack_release(DLManagedTensorVersioned* tensor) { dl::DLPackDeleter{}(tensor); }
void dl_dlpack_release_legacy(DLManagedTensor* tensor) { dl::DLPackDeleter{}(tensor); }

} // extern "C"

#pragma once

#include "dl/npu_export.h"
#include "dl/tensor.h"

#include <filesystem>
#include <memory>
#include <string>

namespace dl::npu {

enum class Performance { latency, throughput };

struct CompileOptions {
    Performance performance = Performance::latency;
    /// 非空路径启用编译缓存；缓存目录由应用放在模型或开发产物目录中。
    std::filesystem::path cache_directory;
    /// 非空时按输入端口顺序给出全部固定形状，在编译前一起应用。
    std::vector<Shape> input_shapes;
};

/// 端口顺序对应请求中的索引；name 是模型中的一个名称，无名称时为空。
/// dtype 描述宿主输入输出的存储类型，shape 描述本次编译的固定形状。
struct Port {
    std::string name;
    DType dtype;
    Shape shape;
};

class Request;

/// 从 ONNX 或 OpenVINO IR 编译 Intel NPU 模型，多个请求共享编译结果。
/// 复制此句柄保留模型身份；每个请求持有模型，使执行资源随请求一起存活。
class DL_NPU_API CompiledModel {
public:
    explicit CompiledModel(const std::filesystem::path& path, CompileOptions options = {});

    const std::vector<Port>& inputs() const;
    const std::vector<Port>& outputs() const;
    std::size_t optimal_request_count() const;
    Request create_request() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Request;
};

/// 一个可复用的推理槽，拥有独立输入、输出及模型内部状态。
/// 每次执行先设置全部输入，再调用 infer 或 start_async；异步完成后读取输出。
/// 多个请求可同时执行，同一个请求由调用方按上述顺序串行使用。
class DL_NPU_API Request {
public:
    ~Request();
    Request(Request&&) noexcept;
    Request& operator=(Request&&) noexcept;
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;

    /// 输入的 dtype 与 shape 对应编译端口；按逻辑顺序复制到请求拥有的缓冲。
    /// CUDA 输入先传到 CPU，非连续输入先打包；返回后原张量可立即释放或更新。
    void set_input(std::size_t index, const Tensor& input);

    void infer();
    void start_async();
    /// wait 传播异步执行结果，ready 查询已提交请求是否完成。
    void wait();
    bool ready();

    /// 在 infer 返回、wait 返回或 ready 为 true 后读取结果。
    /// 每次返回独立的连续 CPU 张量，requires_grad=false，可跨下一次执行保存。
    Tensor output(std::size_t index = 0) const;
    std::vector<Tensor> outputs() const;

    /// 完成后将结果写入已有的连续、可写 CPU 张量，保留其存储和句柄身份。
    /// destination 与输出端口的 dtype、shape 一致且不参与求导，可在连续推理中复用。
    void output_into(std::size_t index, Tensor destination) const;

    /// 将有状态模型的变量恢复到初始值，例如开始处理一个新的独立序列。
    void reset_state();

private:
    explicit Request(const CompiledModel& model);
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class CompiledModel;
};

} // namespace dl::npu

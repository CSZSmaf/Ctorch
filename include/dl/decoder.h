#pragma once

#include "dl/nn.h"

namespace dl::nn {

/// SiLU(gate(x)) * up(x) 再经 down 投影，输入输出的最后一维均为 dimensions。
class DL_API SwiGLU : public UnaryModule {
public:
    SwiGLU(Index dimensions, Index hidden, bool bias = false, TensorOptions options = {});
    Tensor forward(const Tensor& input) override;
private:
    std::shared_ptr<Linear> gate_, up_, down_;
};

/// 冻结基层的独立副本，训练 alpha/rank * up(down(x))；up 从零初始化。
/// merged 返回独立、冻结的 Linear，适合推理；源层和适配器参数保持原值。
class DL_API LoRALinear : public UnaryModule {
public:
    LoRALinear(const Linear& base, Index rank, double alpha = 1.0);
    Tensor forward(const Tensor& input) override;
    std::shared_ptr<Linear> merged() const;
private:
    Tensor weight_;
    std::optional<Tensor> bias_;
    double scale_;
    std::shared_ptr<Linear> down_, up_;
};

/// 频率为 base^(-2i/dimensions)，按相邻特征对旋转；角度表随 buffer 迁移和保存。
class DL_API RotaryEmbedding : public UnaryModule {
public:
    RotaryEmbedding(Index dimensions, Index capacity, double base = 10000.0, TensorOptions options = {});
    Tensor forward(const Tensor& input) override { return forward(input, 0); }
    Tensor forward(const Tensor& input, Index offset);
private:
    explicit RotaryEmbedding(const Tensor& angles);
    Tensor cosine_, sine_;
};

/// 每次推理请求独立持有 K/V，按倒数第二维追加；首批决定其余维度、dtype 和设备。
/// 缓冲按 capacity 一次分配，clear/truncate 复用存储，返回的视图共享该存储。
/// 首次 append 建立张量布局后，即可读取 key/value 视图。
/// 缓存只保存无求导历史的数据，不登记为参数或 buffer；同一缓存串行使用。
class DL_API KVCache {
public:
    explicit KVCache(Index capacity);
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    KVCache(KVCache&&) = default;
    KVCache& operator=(KVCache&&) = default;
    void append(const Tensor& key, const Tensor& value);
    Tensor key() const;
    Tensor value() const;
    Index length() const { return length_; }
    Index capacity() const { return capacity_; }
    void clear() { length_ = 0; }
    void truncate(Index length);
private:
    Index capacity_, length_ = 0;
    std::optional<Tensor> key_, value_;
};

/// 因果自注意力：Q 与 KV 可使用不同头数，投影无 bias，位置采用 RoPE。
class DL_API GroupedQueryAttention : public UnaryModule {
public:
    GroupedQueryAttention(Index dimensions, Index heads, Index kv_heads, Index context,
                          TensorOptions options = {});
    Tensor forward(const Tensor& input) override;
    /// 只计算新 token；offset 来自 cache.length，整个调用不记录梯度。
    Tensor decode(const Tensor& input, KVCache& cache);
private:
    Tensor run(const Tensor& input, KVCache* cache);
    Index dimensions_, heads_, kv_heads_;
    std::shared_ptr<Linear> query_, key_, value_, output_;
    std::shared_ptr<RotaryEmbedding> rotary_;
};

/// PreNorm 层组合 RMSNorm、因果 GQA、SwiGLU 与两次残差相加。
class DL_API TransformerDecoderLayer : public UnaryModule {
public:
    TransformerDecoderLayer(Index dimensions, Index heads, Index kv_heads, Index hidden,
                            Index context, TensorOptions options = {});
    Tensor forward(const Tensor& input) override;
    Tensor decode(const Tensor& input, KVCache& cache);
private:
    std::shared_ptr<GroupedQueryAttention> attention_;
    std::shared_ptr<SwiGLU> feedforward_;
    std::shared_ptr<RMSNorm> attention_norm_, feedforward_norm_;
};

/// 输入输出为 [batch,length,dimensions]；最后一次 RMSNorm 归一化堆叠结果。
/// 每个请求通过 make_cache 创建一组空缓存；各层以同样的 token 分块推进。
class DL_API TransformerDecoder : public UnaryModule {
public:
    TransformerDecoder(Index layers, Index dimensions, Index heads, Index kv_heads, Index hidden,
                       Index context, TensorOptions options = {});
    Tensor forward(const Tensor& input) override;
    std::vector<KVCache> make_cache() const;
    Tensor decode(const Tensor& input, std::span<KVCache> cache);
private:
    Index context_;
    std::vector<std::shared_ptr<TransformerDecoderLayer>> layers_;
    std::shared_ptr<RMSNorm> normalization_;
};

} // namespace dl::nn

"""同一组权重在 PyTorch 与 TensorFlow 中的直接算子实现。"""

import math


class TorchOps:
    def __init__(self, weights, parameters, device):
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.F = torch, F
        self.weights = {name: torch.tensor(value, device=device, requires_grad=name in parameters)
                        for name, value in weights.items()}
        self.device, self.version = device, torch.__version__

    def tensor(self, value): return self.torch.tensor(value, device=self.device)
    def array(self, value): return value.detach().cpu().numpy()
    def transpose(self, x, axes): return x.permute(*axes)
    def relu(self, x): return self.F.relu(x)
    def gelu(self, x): return self.F.gelu(x, approximate='none')
    def linear(self, x, name): return self.F.linear(x, self.weights[name + '.weight'], self.weights.get(name + '.bias'))
    def conv(self, x, name, stride=1, padding=0):
        return self.F.conv2d(x, self.weights[name + '.weight'], self.weights.get(name + '.bias'), stride, padding)
    def batch_norm(self, x, name):
        w = self.weights
        return self.F.batch_norm(x, w[name + '.running_mean'], w[name + '.running_variance'],
                                 w[name + '.weight'], w[name + '.bias'], training=False, eps=1e-5)
    def layer_norm(self, x, name):
        return self.F.layer_norm(x, (x.shape[-1],), self.weights[name + '.weight'], self.weights[name + '.bias'], 1e-5)
    def pool(self, x): return self.F.max_pool2d(x, 2)
    def flatten_image(self, x): return x.flatten(1)
    def average_image(self, x): return x.mean((2, 3))
    def embed(self, x, name): return self.F.embedding(x, self.weights[name])
    def attention(self, q, k, v):
        length = q.shape[-2]
        mask = self.torch.ones((length, length), dtype=self.torch.bool, device=self.device).tril()
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        return self.F.softmax(scores.masked_fill(~mask, -float('inf')), dim=-1) @ v
    def loss(self, logits, targets):
        return self.F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
    def synchronize(self):
        if self.device != 'cpu': self.torch.cuda.synchronize()


class TensorFlowOps:
    def __init__(self, weights, parameters, device):
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        self.tf, self.version = tf, tf.__version__
        # 卷积使用 NHWC/HWIO；只在载入时转换权重，导出梯度时恢复 OIHW。
        self.convolutions = {name for name, value in weights.items() if value.ndim == 4}
        self.weights = {name: tf.Variable(value.transpose(2, 3, 1, 0) if name in self.convolutions else value,
                                          trainable=name in parameters)
                        for name, value in weights.items()}

    def tensor(self, value): return self.tf.convert_to_tensor(value)
    def array(self, value): return self.tf.convert_to_tensor(value).numpy()
    def transpose(self, x, axes): return self.tf.transpose(x, axes)
    def relu(self, x): return self.tf.nn.relu(x)
    def gelu(self, x): return self.tf.nn.gelu(x, approximate=False)
    def linear(self, x, name):
        result = self.tf.linalg.matmul(x, self.weights[name + '.weight'], transpose_b=True)
        bias = self.weights.get(name + '.bias')
        return result if bias is None else result + bias
    def conv(self, x, name, stride=1, padding=0):
        if padding: x = self.tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        result = self.tf.nn.conv2d(x, self.weights[name + '.weight'], strides=[1, stride, stride, 1], padding='VALID')
        bias = self.weights.get(name + '.bias')
        return result if bias is None else result + bias
    def batch_norm(self, x, name):
        w = self.weights
        return self.tf.nn.batch_normalization(x, w[name + '.running_mean'], w[name + '.running_variance'],
                                              w[name + '.bias'], w[name + '.weight'], 1e-5)
    def layer_norm(self, x, name):
        mean, variance = self.tf.nn.moments(x, [-1], keepdims=True)
        return self.tf.nn.batch_normalization(x, mean, variance, self.weights[name + '.bias'],
                                              self.weights[name + '.weight'], 1e-5)
    def pool(self, x): return self.tf.nn.max_pool2d(x, 2, 2, 'VALID')
    def flatten_image(self, x):
        x = self.tf.transpose(x, [0, 3, 1, 2])
        return self.tf.reshape(x, [x.shape[0], -1])
    def average_image(self, x): return self.tf.reduce_mean(x, [1, 2])
    def embed(self, x, name): return self.tf.gather(self.weights[name], x)
    def attention(self, q, k, v):
        length = q.shape[-2]
        mask = self.tf.linalg.band_part(self.tf.ones((length, length), dtype=self.tf.bool), -1, 0)
        scores = self.tf.linalg.matmul(q, k, transpose_b=True) / math.sqrt(q.shape[-1])
        return self.tf.linalg.matmul(self.tf.nn.softmax(self.tf.where(mask, scores, -float('inf')), -1), v)
    def loss(self, logits, targets):
        logits = self.tf.reshape(logits, [-1, logits.shape[-1]])
        targets = self.tf.reshape(targets, [-1])
        valid = targets != -100
        losses = self.tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.tf.boolean_mask(targets, valid), logits=self.tf.boolean_mask(logits, valid))
        return self.tf.reduce_mean(losses)
    def synchronize(self): pass  # 本示例中的 TensorFlow CPU eager 运算同步返回。


def forward(name, x, ops):
    if name == 'mlp':
        x = x.reshape((x.shape[0], -1)) if isinstance(ops, TorchOps) else ops.tf.reshape(x, [x.shape[0], -1])
        return ops.linear(ops.relu(ops.linear(x, 'hidden')), 'output')
    if name == 'lenet':
        x = ops.pool(ops.relu(ops.conv(x, 'conv1')))
        x = ops.pool(ops.relu(ops.conv(x, 'conv2')))
        x = ops.relu(ops.linear(ops.flatten_image(x), 'hidden1'))
        return ops.linear(ops.relu(ops.linear(x, 'hidden2')), 'output')
    if name == 'resnet':
        x = ops.relu(ops.batch_norm(ops.conv(x, 'stem', padding=1), 'stem_norm'))
        for i in range(9):
            prefix, stride = f'block{i}', 2 if i in (3, 6) else 1
            residual = ops.batch_norm(ops.conv(x, prefix + '.shortcut.conv', stride), prefix + '.shortcut.norm') if stride == 2 else x
            y = ops.relu(ops.batch_norm(ops.conv(x, prefix + '.conv1', stride, 1), prefix + '.norm1'))
            x = ops.relu(ops.batch_norm(ops.conv(y, prefix + '.conv2', padding=1), prefix + '.norm2') + residual)
        return ops.linear(ops.average_image(x), 'output')

    # 四层 Pre-LN Transformer，词表投影与输入 Embedding 共用 output.weight。
    reshape = (lambda value, shape: value.reshape(shape)) if isinstance(ops, TorchOps) else ops.tf.reshape
    positions = ops.weights['positions'][:x.shape[1]]
    x = ops.embed(x, 'output.weight') + ops.embed(positions, 'positions_embedding.weight')
    batch, length, width = x.shape
    for i in range(4):
        prefix = f'encoder.layer{i}'
        normalized = ops.layer_norm(x, prefix + '.attention_norm')
        projected = [ops.transpose(reshape(ops.linear(normalized, prefix + '.attention.' + part),
                                           (batch, length, 4, width // 4)), (0, 2, 1, 3)) for part in ('query', 'key', 'value')]
        attended = reshape(ops.transpose(ops.attention(*projected), (0, 2, 1, 3)), (batch, length, width))
        x = x + ops.linear(attended, prefix + '.attention.output')
        normalized = ops.layer_norm(x, prefix + '.feedforward_norm')
        x = x + ops.linear(ops.gelu(ops.linear(normalized, prefix + '.expand')), prefix + '.contract')
    return ops.linear(ops.layer_norm(x, 'encoder.normalization'), 'output')

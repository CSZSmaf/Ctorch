"""读取 Ctorch 项目快照，复算输出、参数梯度、SGD 更新和前向延迟。"""

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from frameworks import TorchOps, TensorFlowOps, forward


def read_archive(path):
    records = {}
    with path.open('rb') as source:
        def integer(): return struct.unpack('<Q', source.read(8))[0]
        if source.read(8) != b'CTORCH\x00\x01':
            raise ValueError('expected a Ctorch v1 tensor archive')
        for _ in range(integer()):
            name = source.read(integer()).decode()
            dtype = np.dtype(['<f4', '<f8', '<i8', '?'][source.read(1)[0]])
            shape = tuple(integer() for _ in range(integer()))
            records[name] = np.frombuffer(source.read(math.prod(shape) * dtype.itemsize), dtype).copy().reshape(shape)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--framework', choices=['pytorch', 'tensorflow'], required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if args.framework == 'tensorflow' and args.device != 'cpu':
        parser.error('this TensorFlow example uses the CPU')
    directory = args.directory
    metadata = json.loads((directory / 'ctorch.json').read_text())
    name = metadata['model']
    weights, expected = read_archive(directory / 'weights.ct'), read_archive(directory / 'results.ct')
    parameters = [key[5:] for key in expected if key.startswith('grad.')]
    backend = TorchOps if args.framework == 'pytorch' else TensorFlowOps
    ops = backend(weights, parameters, args.device)
    x, target = ops.tensor(expected['input']), ops.tensor(expected['target'])
    if args.framework == 'tensorflow' and name in ('lenet', 'resnet'):
        x = ops.transpose(x, (0, 2, 3, 1))

    with ops.tf.GradientTape() if args.framework == 'tensorflow' else contextlib.nullcontext() as tape:
        logits = forward(name, x, ops)
        loss = ops.loss(logits, target)
    if args.framework == 'pytorch':
        loss.backward()
        gradients = [ops.weights[key].grad for key in parameters]
    else:
        gradients = tape.gradient(loss, [ops.weights[key] for key in parameters])

    actual = {'logits': ops.array(logits), 'loss': ops.array(loss)}
    for key, gradient in zip(parameters, gradients):
        values = ops.array(gradient)
        if args.framework == 'tensorflow' and key in ops.convolutions: values = values.transpose(3, 2, 0, 1)
        actual['grad.' + key] = values
        if args.framework == 'pytorch':
            with ops.torch.no_grad(): ops.weights[key].add_(gradient, alpha=-0.01)
        else:
            ops.weights[key].assign_sub(0.01 * ops.tf.convert_to_tensor(gradient))
        values = ops.array(ops.weights[key])
        if args.framework == 'tensorflow' and key in ops.convolutions: values = values.transpose(3, 2, 0, 1)
        actual['updated.' + key] = values

    # 绝对误差衡量偏差大小，逐元素容差同时约束较大数值的相对误差。
    comparisons = []
    for key, value in actual.items():
        difference = np.abs(value.astype(np.float64) - expected[key])
        comparisons.append({'tensor': key, 'max_abs_error': float(difference.max()),
                            'passed': bool(np.allclose(value, expected[key], atol=2e-4, rtol=2e-4))})
    with ops.torch.no_grad() if args.framework == 'pytorch' else contextlib.nullcontext():
        for _ in range(3): forward(name, x, ops)
        ops.synchronize()
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            for _ in range(5): forward(name, x, ops)
            ops.synchronize()
            samples.append((time.perf_counter() - start) * 1000 / 5)
    labels = expected['target'].reshape(-1)
    predicted = np.argmax(actual['logits'], -1).reshape(-1)
    valid = labels != -100
    report = {'model': name, 'framework': args.framework, 'version': ops.version, 'device': args.device,
              'loss': float(actual['loss']), 'accuracy': float(np.mean(predicted[valid] == labels[valid])),
              'forward_ms': statistics.median(samples), 'atol': 2e-4, 'rtol': 2e-4,
              'all_passed': all(item['passed'] for item in comparisons), 'comparisons': comparisons}
    (directory / f'{args.framework}-{args.device}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'comparisons'}))
    if not report['all_passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()

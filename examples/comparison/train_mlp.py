"""使用 Ctorch 导出的初值和批次顺序，在相同 MNIST 样本上训练 MLP。"""

import argparse
import contextlib
import csv
import json
from pathlib import Path
import numpy as np
from compare import read_archive
from frameworks import TorchOps, TensorFlowOps, forward


def mnist(directory, prefix):
    images = np.frombuffer((directory / f'{prefix}-images-idx3-ubyte').read_bytes(), np.uint8, offset=16)
    labels = np.frombuffer((directory / f'{prefix}-labels-idx1-ubyte').read_bytes(), np.uint8, offset=8)
    return images.reshape(-1, 1, 28, 28).astype(np.float32) / 255, labels.astype(np.int64)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset', type=Path)
parser.add_argument('directory', type=Path)
parser.add_argument('--framework', choices=['pytorch', 'tensorflow'], required=True)
args = parser.parse_args()
weights = read_archive(args.directory / 'initial.ct')
order = read_archive(args.directory / 'order.ct')['indices']
ops = (TorchOps if args.framework == 'pytorch' else TensorFlowOps)(weights, list(weights), 'cpu')
train_x, train_y = mnist(args.dataset, 'train')
with (args.directory / f'{args.framework}-curve.csv').open('w', newline='') as file:
    curve = csv.writer(file)
    curve.writerow(['step', 'loss'])
    for step, selected in enumerate(order, 1):
        x, target = ops.tensor(train_x[selected]), ops.tensor(train_y[selected])
        with ops.tf.GradientTape() if args.framework == 'tensorflow' else contextlib.nullcontext() as tape:
            loss = ops.loss(forward('mlp', x, ops), target)
        if args.framework == 'pytorch':
            loss.backward()
            with ops.torch.no_grad():
                for value in ops.weights.values():
                    value.add_(value.grad, alpha=-0.1)
                    value.grad = None
        else:
            gradients = tape.gradient(loss, list(ops.weights.values()))
            for value, gradient in zip(ops.weights.values(), gradients): value.assign_sub(0.1 * gradient)
        curve.writerow([step, float(ops.array(loss))])

test_x, test_y = mnist(args.dataset, 't10k')
total_loss, correct = 0.0, 0
with ops.torch.no_grad() if args.framework == 'pytorch' else contextlib.nullcontext():
    for start in range(0, len(test_y), 256):
        labels = test_y[start:start + 256]
        logits = forward('mlp', ops.tensor(test_x[start:start + 256]), ops)
        total_loss += float(ops.array(ops.loss(logits, ops.tensor(labels)))) * len(labels)
        correct += int(np.sum(np.argmax(ops.array(logits), axis=1) == labels))
expected = read_archive(args.directory / 'ctorch-final.ct')
error = max(float(np.max(np.abs(ops.array(value) - expected[name]))) for name, value in ops.weights.items())
report = {'framework': args.framework, 'version': ops.version, 'steps': len(order),
          'test_loss': total_loss / len(test_y), 'test_accuracy': correct / len(test_y),
          'max_weight_difference': error}
(args.directory / f'{args.framework}-training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))

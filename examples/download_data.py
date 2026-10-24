"""将项目示例使用的数据下载到指定目录。仅在直接运行此脚本时下载。"""

import argparse
import gzip
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlretrieve

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset', choices=['mnist', 'cifar10', 'shakespeare'])
parser.add_argument('directory', type=Path)
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)

if args.dataset == 'mnist':
    base = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    for name in ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']:
        path = args.directory / (name + '.gz')
        urlretrieve(base + path.name, path)
        with gzip.open(path, 'rb') as source, (args.directory / name).open('wb') as target:
            shutil.copyfileobj(source, target)
        print(name)
elif args.dataset == 'cifar10':
    path = args.directory / 'cifar-10-binary.tar.gz'
    urlretrieve('https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz', path)
    with tarfile.open(path) as archive:
        archive.extractall(args.directory, filter='data')
    print(args.directory / 'cifar-10-batches-bin')
else:
    path = args.directory / 'input.txt'
    urlretrieve('https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt', path)
    print(path)

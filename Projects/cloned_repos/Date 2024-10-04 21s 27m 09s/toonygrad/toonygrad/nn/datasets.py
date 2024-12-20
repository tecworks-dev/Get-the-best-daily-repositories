from toonygrad.tensor import Tensor
from toonygrad.helpers import fetch
from toonygrad.nn.state import tar_extract

def _mnist(file): return Tensor(fetch("https://storage.googleapis.com/cvdf-datasets/mnist/"+file, gunzip=True))
def mnist(device=None):
  return _mnist("train-images-idx3-ubyte.gz")[0x10:].reshape(-1,1,28,28).to(device), _mnist("train-labels-idx1-ubyte.gz")[8:].to(device), \
         _mnist("t10k-images-idx3-ubyte.gz")[0x10:].reshape(-1,1,28,28).to(device), _mnist("t10k-labels-idx1-ubyte.gz")[8:].to(device)

def cifar(device=None):
  tt = tar_extract(fetch('https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz', gunzip=True))
  train = Tensor.cat(*[tt[f"cifar-10-batches-bin/data_batch_{i}.bin"].reshape(-1, 3073).to(device) for i in range(1,6)])
  test = tt["cifar-10-batches-bin/test_batch.bin"].reshape(-1, 3073).to(device)
  return train[:, 1:].reshape(-1,3,32,32), train[:, 0], test[:, 1:].reshape(-1,3,32,32), test[:, 0]

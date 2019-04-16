'''
We'll feed inputs into our neural network in batches.
Here are some tools for iterating over data in batches.
'''
import numpy as np
from tensor import Tensor
from typing import Iterator, NamedTuple

Batch = NamedTuple("Batch",[("inputs", Tensor),("targets", Tensor)])

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator:
        raise NotImplementedError

class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator:
        starts = np.arange(0,len(inputs), self.batch_size) #start, end, step
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)




'''
A loss function measures how good our prediction are,
we can use this to adjust the parameters of our network
'''

import numpy as np
from tensor import Tensor
#from tensor.py import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        '''

        :param predicted:
        :param actual:
        :return:
        '''
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        '''
        A vector of partial derivatives
        :param predicted:
        :param actual:
        :return:
        '''
        raise NotImplementedError

class MES(Loss):
    '''
    MES is mean squared errot, although we're just going to
    do total squared error
    '''
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        '''

        :param predicted:
        :param actual:
        :return:
        '''
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        '''
        A vector of partial derivatives
        :param predicted:
        :param actual:
        :return:
        '''
        return 2 * (predicted - actual)
'''
Use an optimizer to ajust parameters
of our network based on the gradients computed
during backpropagation
'''
from nn import NeuralNet

class Optimizer:
    def step(self,net: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self,net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad            #adjust params by declining learning rate * gradient





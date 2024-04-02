import numpy as np

from variable import Variable
from utils import as_array


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


class Function:
    """
    Function class

    Methods of mapping relationships between variables are implemented by inheriting from the class through instances
    of the class.
    """

    def __call__(self, var):
        output = Variable(as_array(self.forward(var.data)))
        output.set_creator(self)
        self.input = var
        self.output = output
        return output

    def forward(self, data):
        """Forward propagation"""
        raise NotImplementedError()

    def backward(self, grad):
        """Backward propagation"""
        raise NotImplementedError()


class Square(Function):
    """
    Square function

    This method calculates the square of the input data.
    """

    def forward(self, x):
        return x ** 2

    def backward(self, grad):
        return 2 * self.input.data * grad


class Exp(Function):
    """
    Exponential function

    This method calculates the exponential of the input data
    """

    def forward(self, x):
        return np.exp(x)

    def backward(self, grad):
        return np.exp(self.input.data) * grad

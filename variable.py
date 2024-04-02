from collections import deque

import numpy as np

from utils import as_array


class Variable:
    """
    Variable class

    Used to store data.

    Attributes:
        data: Stored Data
        grad: Gradient
        creator: The function that produces the variable
    """

    def __init__(self, data):
        """Constructor"""
        if data is not None:
            if not isinstance(data, np.ndarray):
                as_array(data)

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        """Set the creator function."""
        self.creator = func

    def backward(self):
        """Backpropagation gradients."""
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = deque([self.creator])
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)

import numpy as np
import math as math


class Scalar:
    def __init__(self, data, name='', _children=(), _op='', ):
        # value data variables
        self.data = data
        self.grad = 0
        self.name = name

        # Internal variables
        self._children = _children
        self._backward = lambda: None

    def __repr__(self):
        return f"Scalar: {self.name}, Data = {self.data}, grad = {self.grad}"

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = Scalar(data=self.data + other.data, _op='+', _children=(self, other))

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
            out._backward = _backward

        return out

    def __sub__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = Scalar(data=self.data - other.data, _op='-', _children=(self, other))

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
            out._backward = _backward

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = Scalar(data=self.data * other.data, _op='*', _children=(self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            out._backward = _backward

        return out

    def __pow__(self, other):
        """
        self^other
        for any power not only for exp.
        :param other:
        :return: the power to any number (int,float)
        """
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = Scalar(data=self.data ** other.data, _op='^', _children=(self, other))

        def _backward():
            self.grad += (other.data * self.data ** (other - 1)) * out.grad
            other.grad += (self.data ** other.data) * np.log(self.data) * out.grad

            out._backward = _backward

        return out

    def exp(self):
        """
        exp func.
        :return: Return e raised to the power of {self.data}
        """
        out = Scalar(data=math.exp(self.data), _op='e^', _children=(self,))

        def _backward():
            self.grad += out.data * out.grad
            out._backward = _backward

        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = self * other ** -1
        return out

    def __radd__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = self + other
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = self + other
        return out
    def __rmul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = self * other
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = other * self ** -1
        return out

    def __rpow__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other)
        out = other ** self
        return out

    def __neg__(self):
        return -1 * self

    def tanh(self):
        t = np.tanh(self.data)
        out = Scalar(data=t, _op='tanh', _children=(self,))

        def _backwards():
            self.grad += 1 - t ** 2 * out.grad
            out._backward = _backwards

        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(self.data))
        out = Scalar(data=s, _op='Sigmoid', _children=(self,))

        def _backward():
            self.grad += s * (1 - s) * out.grad
            out._backward = _backward

        return out

    def relu(self):
        r = float(self.data > 0)
        out = Scalar(data=r * self.data, _op="Relu", _children=(self,))

        def _backward():
            self.grad += r * out.grad
            out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for scalar in v._children:
                    build(scalar)
                topo.append(v)

        build(self)

        self.grad = 1
        for node in topo:
            node._backward()

import random

from engine import Scalar


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, activation='linear'):
        self.w = [Scalar(data=random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Scalar(data=0)
        self.activation = activation

    def __call__(self, x):
        linear = sum([xi * wi for xi, wi in zip(x, self.w)], self.b)
        if self.activation == 'relu':
            linear = linear.relu()
        elif self.activation == 'sigmoid':
            linear = linear.sigmoid()
        elif self.activation == 'tanh':
            linear = linear.tanh()

        return linear

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Activation = {self.activation} Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.nuerons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.nuerons]
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        return f"Layer of [{','.join([str(n) for n in self.nuerons])}]"

    def parameters(self):
        parameters = []
        for n in self.nuerons:
            parameters += n.parameters()
        return parameters


class MLP(Module):
    def __init__(self, nin, nouts):
        size = [nin] + nouts
        self.layers = []
        for i in range(len(size)):
            if i == len(size) - 1:
                self.layers.append(Layer(size[i], size[i + 1], activation='linear'))
            else:
                self.layers.append(Layer(size[i], size[i + 1], activation='tanh'))

    def __repr__(self):
        return f"MlP:\n {','.join([str(layer) for layer in self.layers])}"

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters += layer.parameters()
        return parameters

"""
Core module for the library.
"""
import math


class Value:

    def __init__(self, data, name: str = '', operator: str = '', children: list = []):
        self.data = data
        self.name = name
        self.operator = operator
        self.children = children
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self):
        return f'Value(data={self.data:0.3f})'

    def __add__(self, other):
        if type(other) in (float, int):
            other = Value(other)
        out = Value(self.data + other.data, operator='+',
                    children=[self, other])

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        self._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if type(other) in (float, int):
            other = Value(other)
        out = Value(self.data * other.data, operator='*',
                    children=[self, other])

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        self._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __truediv__(self, other):
        return self * other ** -1

    def __pow__(self, other):
        if type(other) not in (float, int):
            raise ValueError('Only supports float or int')
        out = Value(self.data ** other, operator=f'**{other}',
                    children=[self])

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        self._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t,
                    operator='tanh',
                    children=[self])

        def _backward():
            self.grad = (1 - t**2) * out.grad

        self._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0,
                    operator='relu',
                    children=[self])

        def _backward():
            self.grad = (1 if self.data > 0 else 0) * out.grad

        self._backward = _backward
        return out

    def backward(self):
        """Backward pass."""
        self.grad = 1
        visited = set()
        queue = [self]

        while True:
            if not queue:
                break

            value = queue.pop(0)
            if value in visited:
                continue

            value._backward()
            visited.add(value)
            queue.extend(value.children)

    def zero_grad(self):
        """Resets the gradients for all the nodes in a graph."""
        self.grad = 0
        visited = set()
        queue = [self]

        while True:
            if not queue:
                break

            value = queue.pop(0)
            if value in visited:
                continue

            value.grad = 0
            visited.add(value)
            queue.extend(value.children)

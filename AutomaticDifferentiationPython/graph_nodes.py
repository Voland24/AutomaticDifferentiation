import graph
import numpy as np

class Node:
    """
    Used for checking if an object is a Graph Node or not
    """

    def __init__(self):
        pass


class PlaceHolder(Node):
    """
    Placeholder holds a node in the computational graph and awaits a value to be assigned to it at computation time
    """

    count = 0

    def __init__(self, name, dtype="float"):
        graph._g.placeholders.add(self)
        self.value = None
        self.gradient = None
        self.name = f"Plc/{PlaceHolder.count}" if name is None else name
        PlaceHolder.count += 1

    def __repr__(self):
        return f"Placeholder: name:{self.name}, values:{self.value}"


class Constant(Node):
    """
    A constant in the computational graph
    """

    count = 0

    def __init__(self, value, name=None):
        graph._g.constants.add(self)
        self._value = value
        self.gradient = None
        self.name = f"Const/{Constant.count}" if name is None else name
        Constant.count += 1

    def __repr__(self):
        return f"Constant: name{self.name}, value:{self.value}"

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self):
        raise ValueError("Cannot reassign constants")
        self._value = value
        self.gradient = None
        self.name = f"Const/{Constant.count}" if name is None else name
        Constant.count += 1


class Variable(Node):
    """
    A variable in the computational graph. These are automatically tracked during graph computation
    """

    count = 0

    def __init__(self, value, name=None):
        super().__init__()
        graph._g.variables.add(self)
        self.value = value
        self.gradient = None
        self.name = f"Var/{Variable.count}" if name is None else name
        Variable.count += 1

    def __repr__(self):
        return f"Variable: name:{self.name}, value: {self.value}"



class Operator(Node):
    """
    This is a node in the computational graph
    """

    def __init__(self, name="Operator"):
        graph._g.operators.add(self)
        self.value = None
        self.inputs = []
        self.gradient = None
        self.name = name

    def __repr__(self):
        return f"Operator: name{self.name}"


class add(Operator):
    count = 0
    """Binary addition operation."""

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'add/{add.count}' if name is None else name
        add.count += 1

    def forward(self, a, b):
        return a + b

    def backward(self, a, b, dout):
        return dout, dout


class multiply(Operator):
    count = 0
    """Binary multiplication operation."""

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'mul/{multiply.count}' if name is None else name
        multiply.count += 1

    def forward(self, a, b):
        return a * b

    def backward(self, a, b, dout):
        return dout * b, dout * a


class divide(Operator):
    count = 0
    """Binary division operation."""

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'div/{divide.count}' if name is None else name
        divide.count += 1

    def forward(self, a, b):
        return a / b

    def backward(self, a, b, dout):
        return dout / b, dout * a / np.power(b, 2)


class power(Operator):
    count = 0
    """Binary exponentiation operation."""

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'pow/{power.count}' if name is None else name
        power.count += 1

    def forward(self, a, b):
        return np.power(a, b)

    def backward(self, a, b, dout):
        return dout * b * np.power(a, (b - 1)), dout * np.log(a) * np.power(a, b)


class matmul(Operator):
    count = 0
    """Binary multiplication operation."""

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'matmul/{matmul.count}' if name is None else name
        matmul.count += 1

    def forward(self, a, b):
        return a @ b

    def backward(self, a, b, dout):
        return dout @ b.T, a.T @ dout


def node_wrapper(func, self, other):
    if isinstance(other, Node):
        return func(self, other)
    if isinstance(other, float) or isinstance(other, int):
        return func(self, Constant(other))
    raise TypeError("Incompatible types.")


Node.__add__ = lambda self, other: node_wrapper(add, self, other)
Node.__mul__ = lambda self, other: node_wrapper(multiply, self, other)
Node.__div__ = lambda self, other: node_wrapper(divide, self, other)
Node.__neg__ = lambda self: node_wrapper(multiply, self, Constant(-1))
Node.__pow__ = lambda self, other: node_wrapper(power, self, other)
Node.__matmul__ = lambda self, other: node_wrapper(matmul, self, other)
import numpy as np
from graph_nodes import Node
from graph_nodes import Operator, PlaceHolder


class Graph:
    """
    Computational graph object used for evaluating functions and their derivatives.
    """

    def __init__(self):
        self.operators = set()
        self.constants = set()
        self.variables = set()
        self.placeholders = set()
        global _g
        _g = self

    def reset_counts(self, root):
        if hasattr(root, "count"):
            root.count = 0
        else:
            for child in root.__subclasses__():
                self.reset_counts(child)

    def reset_session(self):
        try:
            del _g
        except:
            pass
        self.reset_counts(Node)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset_session()


def forward_pass(order, feed_dict={}):
    """
    Performs the forward pass, returning the output of the graph
    :param order: topologically sorted array of nodes
    :param feed_dict: a dict of values for placeholders
    :return: final result of graph execution and edits the graph to fill in the current values
    """

    for node in order:
        if isinstance(node, PlaceHolder):
            node.value = feed_dict[node.name]
        elif isinstance(node, Operator):
            node.value = node.forward(*[prev_node.value for prev_node in node.inputs])

    return order[-1].value

def backward_pass(order):
    """
    Performs the backward pass to retrieve the gradients.
    :param order: topologically ordered array of graph nodes, gradient of the final node, the output node is by deafult 1.
    :return: gradient of nodes listed in the same order as the input arguments
    """
    vis = set()
    order[-1].gradient = 1
    for node in reversed(order):
        if isinstance(node, Operator):
            inputs = node.inputs
            grads = node.backward(*[x.value for x in inputs], dout=node.gradient)
            for inp, grad in zip(inputs, grads):
                if inp not in vis:
                    inp.gradient = grad
                else:
                    inp.gradient += grad
                vis.add(inp)
    return [node.gradient for node in order]


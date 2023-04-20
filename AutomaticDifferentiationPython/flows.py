
import graph
from graph_nodes import Operator, PlaceHolder
def topological_sort(head_node = None, comp_graph = graph._g):
    """
    Performs the topological ordering of the computational graph, i.e. orders all the nodes prior to and including the head_node
    :param head_node: last node in forward pass, the output of the graph
    :param graph: computational graph to be calculated
    :return: a sorted array of graph nodes
    """

    vis = set()
    ordering = []
    def _dfs(node):
        if node not in vis:
            vis.add(node)
            if isinstance(node, Operator):
                for input_node in node.inputs:
                    _dfs(input_node)
            ordering.append(node)

    if head_node is None:
        for node in comp_graph.operators:
            _dfs(node)
    else:
        _dfs(head_node)


    return ordering

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
            for inp, grad in zip(input, grads):
                if inp not in vis:
                    inp.gradient = grad
                else:
                    inp.gradient += grad

    return [node.gradient for node in order]

import graph
from graph_nodes import Operator


class GraphSorter:

    def __init__(self):
        self.graph = graph._g

    def set_graph(self, graph = None):
        if graph is None:
            self.graph = graph._g
        else:
            self.graph = graph

    def topological_sort(self, head_node=None):
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
            for node in self.graph.operators:
                _dfs(node)
        else:
            _dfs(head_node)

        return ordering
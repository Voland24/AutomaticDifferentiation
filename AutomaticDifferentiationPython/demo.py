from graph import Graph, forward_pass, backward_pass
from graph_nodes import Variable, Constant
from topological_sort import GraphSorter


with Graph() as g:
    x = Variable(1.3)
    y = Variable(0.9)
    z = x*y + 5


print(g.variables)
print(g.operators)
print(g.constants)

# take a simple function f(x,y:c) = c * (x*y + c) + x
#find its gradients with respect to x and y, c is a constant

val1 = 0.9
val2 = 0.4
val3 = 1.3
with Graph() as g:
    sorter = GraphSorter()
    x = Variable(val1, name='x')
    y = Variable(val2, name='y')
    c = Constant(val3, name='c')
    z = (x*y + c)*c + x

    sorter.set_graph(z)
    order = sorter.topological_sort(z)
    res = forward_pass(order)
    grads = backward_pass(order)

    print('Nodes in order of appereance')
    [print(node) for node in order]

    print('-'*10)
    print(f'Forward pass expected value : {(val1*val2+val3)*val3+val1}')
    print(f'Graph calculated value : {res}')

    print('-'*10)
    print('Partial derivatives per variable')
    dzdx_node = [a for a in order if a.name=='x'][0]
    dzdy_node = [a for a in order if a.name == 'y'][0]
    dzdc_node = [a for a in order if a.name == 'c'][0]
    print(f"dz/dx expected = {val3 * val2 + 1}")
    print(f"dz/dx computed = {dzdx_node.gradient}")
    print(f"dz/dy expected = {val1 * val3}")
    print(f"dz/dy computed = {dzdy_node.gradient}")
    print(f"dz/dc expected = {val1 * val2 + 2 * val3}")
    print(f"dz/dc computed = {dzdc_node.gradient}")

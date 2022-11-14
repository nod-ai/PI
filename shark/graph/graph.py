import inspect
import os

from python_graphs import program_graph
from python_graphs import program_graph_graphviz
from python_graphs import program_graph_test_components as tc

os.makedirs('out', exist_ok=True)

# For each function in program_graph_test_components.py, visualize its
# program graph. Save the results in the output directory.
for name, fn in inspect.getmembers(tc, predicate=inspect.isfunction):
    path = f'out/{name}-program-graph.png'
    graph = program_graph.get_program_graph(fn)
    program_graph_graphviz.render(graph, path=path)
print('Done. See the `out` directory for the results.')
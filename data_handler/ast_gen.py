from pycparser import c_parser, c_ast
import json
import json
import networkx as nx
import matplotlib.pyplot as plt

# C 코드 파싱
parser = c_parser.CParser()
source_code = '''
int main() {
    printf("Hello World!");
}
'''
ast = parser.parse(source_code)

# AST 시각화
def visualize_ast(node, parent=None):
    G.add_node(node)
    if parent is not None:
        G.add_edge(parent, node)
    for _, child in node.children():
        visualize_ast(child, node)

G = nx.DiGraph()
visualize_ast(ast)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()

from phply.phpparse import make_parser
from phply import phplex
from data.wirecaml_utils.my_php_listener import MyPHPListener
from data.wirecaml_utils.phptraverser import php_traverser
from data.preprocessing import map_tokens
from lxml import etree
import torch
import numpy as np
from data.preprocessing import sub_tokens
from data.staticfg.builder import CFGBuilder
import pickle
import matplotlib.pyplot
from torch_geometric.data import Data

import networkx as nx

def process(line,allvars):
    lexer = phplex.lexer.clone()
    code_tokens = []
    lexer.input("<?php " + line)
    while True:
        tok = lexer.token()
        if not tok:
            break
        if tok in ignore:
            continue
        tok = sub_tokens.sub_token(tok,None,allvars)
        code_tokens.append(tok)
    return map_tokens.tokenise(code_tokens)

def process_func_and_tainted(node):
    out = []

    tokens = sub_tokens.getReplacedTokensLower()
    funcs = node.stmt_funcs.union(node.stmt_vars)
    funcs = [func.lower() for func in funcs]
    for token in tokens:
        if token in funcs:
            out.append(1)
        else:
            out.append(0)

    if node.is_tainted():
        out.append(1)
    else:
        out.append(0)

    return out

def traverse(nodes):
    new_edges = []
    for node in nodes:
        deps = node.get_node_deps()
        index = nodes.index(node)
        for dep in deps:
            dep_i = nodes.index(dep)
            new_edges.append([dep_i + 1,index + 1])

    return new_edges

def get_tokens(data):
    lexer2 = phplex.lexer.clone()
    lexer2.input(data)
    codetokens = []
    while True:
        tok = lexer2.token()
        if not tok:
            break
        if tok in ignore:
            continue
        tok = sub_tokens.sub_token(tok)
        codetokens.append(tok)
    return codetokens

data_ls = []

data_un = set()
data_no_flaw = set()
data_flaw = set()
no_flaws = 0
no = 0
i = 1


def get_control_deps(cfg):
    new_graph = nx.DiGraph()
    new_graph.add_nodes_from(cfg.nodes)
    edges = [(j, i) for (i, j) in cfg.edges]
    new_graph.add_edges_from(edges)
    doms = nx.algorithms.dominance.dominance_frontiers(new_graph, list(new_graph.nodes)[-1])
    edges = []
    for node in cfg.nodes:
        if len(doms[node]) == 0:
            edges.append([0,list(cfg.nodes).index(node) + 1])
            continue
        for s in doms[node]:
            #shift as start node added at start of list of nodes
            edges.append([list(cfg.nodes).index(s) + 1,list(cfg.nodes).index(node) + 1])
    return edges


for type in ['SQLi','XSS', 'Command injection']:
    print(type)
    dir = "./data/SARD/" + type
    tree = etree.parse(dir + "/manifest.xml")
    ignore = {'WHITESPACE', 'OPEN_TAG', 'CLOSE_TAG'}
    for file in tree.findall('testcase/file'):
        p = file.get('path')
        file_data = []

        if not ((type == 'XSS' and p.startswith('CWE_79'))
            or (type == 'SQLi' and p.startswith('CWE_89'))
            or (type == 'Command injection' and p.startswith('CWE_78'))):
            continue
        lexer = phplex.lexer.clone()
        parser = make_parser()
        file_path = dir + '/' + p
        with open(file_path, "r") as myfile:
            data = myfile.read()
            nodes = parser.parse(data, lexer=lexer, tracking=True, debug=False)
            listener = MyPHPListener(name=file)

            codetokens = get_tokens(data)
            if ",".join(codetokens) in data_un:
                continue
            php_traverser.traverse(nodes, listener)

            cfg = listener.get_graph()
            allvars = []
            for node in list(cfg.nodes):
                for var in node.stmt_vars:
                    if var not in allvars:
                        allvars.append(var)
            control_edges = get_control_deps(cfg)
            edges = [[list(cfg.nodes).index(i) +1,list(cfg.nodes).index(j)+1] for (i,j) in cfg.edges]
            old_edges = edges
            edges = traverse(list(cfg.nodes))
            edges.extend(control_edges)
            # edges.extend(old_edges)
            edges = torch.tensor(edges, dtype=torch.long)
            graph_nodes = torch.tensor([map_tokens.tokenise(["START"])] + [process(node.text,list(allvars)) for node in list(cfg.nodes)])
            flaw = file.find('flaw')
            if flaw is not None:
                if ",".join(codetokens) + str(i) not in data_flaw and ",".join(codetokens) not in data_un:
                    data = Data(x=graph_nodes, edge_index=edges.t().contiguous(),y=i)
                    data_un.add(",".join(codetokens))
                    data_flaw.add(",".join(codetokens) + str(i))
                    # number here as it won't pickle as a Data object otherwise
                    data_ls.append([data,1])
                    no_flaws += 1
            else:
                if ",".join(codetokens) not in data_no_flaw and ",".join(codetokens) not in data_un:
                    data = Data(x=graph_nodes, edge_index=edges.t().contiguous(), y=0)
                    data_un.add(",".join(codetokens))
                    data_no_flaw.add(",".join(codetokens))
                    data_ls.append([data, 0])
            # print(no_flaws)
        no += 1

    i+=1
    print(len(data_un))
    print(len(data_flaw))
    print(len(data_no_flaw))
    data_flaw = set()
    data_no_flaw = set()
# remember this is write
with open('./sard_multi_replace_tokens_PDG_token.pkl', 'wb') as f:
   pickle.dump(data_ls, f)

print(no_flaws)

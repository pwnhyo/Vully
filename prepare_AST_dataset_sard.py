from phply.phpparse import make_parser
from phply import phplex
from data.wirecaml_utils.my_php_listener import MyPHPListener
from data.wirecaml_utils.phptraverser import php_traverser
from data.preprocessing import LabelEncodeAST
from lxml import etree
import torch
import ast
from data.preprocessing import sub_tokens
from data.staticfg.builder import CFGBuilder
import pickle
import matplotlib.pyplot
from torch_geometric.data import Data

import networkx as nx
import numpy as np

def process(label):
    sub = label.__class__.__name__
    print(label)
    if sub is 'FunctionCall':
        sub = sub_tokens.sub_token_ast(label.name)
    print(sub)
    ret = LabelEncodeAST.encode(sub)
    if ret == 0:
        print("Unknown type")
        print(label.__class__.__name__)
        print(label)
        exit(1)
    return ret

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

data_ls = []
no_flaws = 0
data_un = set()
data_no_flaw = set()
data_flaw = set()
i = 1
no = 0
for type in ['SQLi', 'XSS', 'Command injection']:
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

            cons = {}
            allnodes = ["BLOCK_ENTRY"] #block of lines
            cons["BLOCK_ENTRY"] = nodes
            def traverse(node):
                allnodes.append(node)
                cons[str(node)] = []
                index = str(node)
                if hasattr(node, 'node'):
                    cons[index].append(node.node)
                    traverse(node.node)
                if hasattr(node, 'expr'):
                    cons[index].append(node.expr)
                    traverse(node.expr)
                if hasattr(node, 'left'):
                    cons[index].append(node.left)
                    traverse(node.left)
                if hasattr(node, 'right'):
                    cons[index].append(node.right)
                    traverse(node.right)
                if hasattr(node, 'params'):
                    for param in node.params:
                        cons[index].append(param)
                        traverse(param)
                if hasattr(node, 'nodes'):
                    for nod in node.nodes:
                        cons[index].append(nod)
                        traverse(nod)
                if hasattr(node, 'elseifs'):
                    for elseif in node.elseifs:
                        cons[index].append(elseif)
                        traverse(elseif.node)
                if hasattr(node, 'else_'):
                    if node.else_ is None:
                        return
                    cons[index].append(node.else_.node)
                    traverse(node.else_.node)


            for node in nodes:
                traverse(node)
            edges = []
            for node in allnodes:
                index_parent = allnodes.index(node)
                for child in cons[str(node)]:
                    index_child = allnodes.index(child)
                    edges.append([index_parent,index_child])
            edges = torch.tensor(edges,  dtype=torch.long)

            graph_nodes = torch.tensor([process(node) for node in allnodes])

            codetokens = ",".join([str(n) for n in graph_nodes])
            flaw = file.find('flaw')
            if flaw is not None:
                if codetokens + str(i) not in data_flaw and codetokens not in data_un:
                    data = Data(x=graph_nodes, edge_index=edges.t().contiguous(),y=i)
                    data_un.add(codetokens)
                    data_flaw.add(codetokens + str(i))
                    # number here as it won't pickle as a Data object otherwise
                    data_ls.append([data,1])
                    no_flaws += 1
            else:
                if codetokens not in data_no_flaw and codetokens not in data_un:
                    data = Data(x=graph_nodes, edge_index=edges.t().contiguous(), y=0)
                    data_un.add(codetokens)
                    data_no_flaw.add(codetokens)
                    data_ls.append([data,0])
            no += 1

    i+=1
    #print(len(data_un))
    #print(len(data_flaw))
    #print(len(data_no_flaw))
    data_flaw = set()
    data_no_flaw = set()
    # data_un = set()

with open('sard_multi_replace_tokens_AST_token.pkl', 'wb') as f:
    pickle.dump(data_ls, f)

print(no_flaws)

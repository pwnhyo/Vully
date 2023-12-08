from phply.phpparse import make_parser
from phply import phplex
from data.wirecaml_utils.my_php_listener import MyPHPListener
from data.wirecaml_utils.phptraverser import php_traverser
from data.preprocessing import map_tokens
from lxml import etree
import torch
import ast
from data.preprocessing import sub_tokens
from data.staticfg.builder import CFGBuilder
import pickle
import matplotlib.pyplot
import numpy as np
from torch_geometric.data import Data

import networkx as nx
import numpy as np

import os

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

max_len = 100
def process_char(line):
    chars = list(line)
    ret = []
    i = 0
    for c in chars:
        if i < max_len:
            ret.append(ord(c))
        else:
            break
        i += 1
    if i < max_len:
        for j in range(max_len - i):
            ret.append(0)
    if len(ret) != 100:
        print("error")
        print(len(ret))
        print(ret)
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
        # uncomment if using windows and hitting path length limit
        # file_path = os.path.abspath(dir + '/' + p)
        # if len(file_path)> 255:
        #     file_path = "\\\\?\\" + file_path
        #     file_path = file_path.encode()
        with open(file_path, "r") as myfile:
            data = myfile.read()
            nodes = parser.parse(data, lexer=lexer, tracking=True, debug=False)
            listener = MyPHPListener(name=file)


            codetokens = get_tokens(data)
            if ",".join(codetokens) in data_un:
                continue

            php_traverser.traverse(nodes, listener)

            cfg = listener.get_graph()
            edges = [[list(cfg.nodes).index(i),list(cfg.nodes).index(j)] for (i,j) in cfg.edges]
            edges = torch.tensor(edges,  dtype=torch.long)
            allvars = []
            for node in list(cfg.nodes):
                for var in node.stmt_vars:
                    if var not in allvars:
                        allvars.append(var)
            graph_nodes = [process(node.text,list(allvars)) for node in list(cfg.nodes)]
            flaw = file.find('flaw')
            if flaw is not None:
                if ",".join(codetokens) + str(i) not in data_flaw and ",".join(codetokens) not in data_un:
                    data = Data(x=torch.tensor(graph_nodes), edge_index=edges.t().contiguous(),y=i)
                    data_un.add(",".join(codetokens))
                    data_flaw.add(",".join(codetokens) + str(i))
                    # number here as it won't pickle as a Data object otherwise
                    data_ls.append([data,1])
                    no_flaws += 1
            else:
                if ",".join(codetokens) not in data_no_flaw and ",".join(codetokens) not in data_un:
                    data = Data(x=torch.tensor(graph_nodes), edge_index=edges.t().contiguous(), y=0)
                    data_un.add(",".join(codetokens))
                    data_no_flaw.add(",".join(codetokens))
                    data_ls.append([data,0])

            no += 1
    i+=1
    # with open('split_data/sard_replace_tokens_graph_no_dup_' + type + '.pkl', 'wb') as f:
    #     pickle.dump(data_ls, f)
    print(len(data_un))
    print(len(data_flaw))
    print(len(data_no_flaw))
    data_flaw = set()
    data_no_flaw = set()
    # data_ls = []
    # data_un = set()
# remember this is write

with open('sard_multi_replace_tokens_CFG_token.pkl', 'wb') as f:
    pickle.dump(data_ls, f)

print(no)
print(no_flaws)

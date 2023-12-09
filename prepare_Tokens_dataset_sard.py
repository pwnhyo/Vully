import random
import signal
import pandas as pd
from phply import phplex
from lxml import etree
from data.preprocessing import sub_tokens
import pickle

data_ls = []
data_un = set()
data_no_flaw = set()
data_flaw = set()
no_flaws = 0
i = 1 # was 1
for type in ['SQLi', 'XSS', 'Command injection']:
    no_flaws_cur = 0
    no_cur = 0
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
        file_path = dir + '/' + p
        code_tokens = []
        codetokens = []
        allvar = []
        with open(file_path, "r") as myfile:
            data = myfile.read()
            lexer.input(data)
            while True:
                tok = lexer.token()
                if not tok:
                    break
                if tok in ignore:
                    continue
                tok1 = sub_tokens.sub_token(tok,None,allvar)
                allvar = list(allvar)
                code_tokens.append(tok1)
                codetokens.append(sub_tokens.sub_token(tok))
        flaw = file.find('flaw')
        if flaw is not None:
            if ",".join(codetokens) + str(i) not in data_flaw and ",".join(codetokens) not in data_un:
                data_ls.append([code_tokens,i])
                data_un.add(",".join(codetokens))
                data_flaw.add(",".join(codetokens) + str(i))
                # print(file_path)
            no_flaws += 1
            no_flaws_cur += 1
        else:
            if ",".join(codetokens) not in data_no_flaw and ",".join(codetokens) not in data_un:
                data_ls.append([code_tokens,0])
                data_un.add(",".join(codetokens))
                data_no_flaw.add(",".join(codetokens))
                # print(file_path)
            no_cur += 1
    i+=1
    print(no_flaws_cur)
    print(no_cur)
    print(len(data_un))
    print(len(data_flaw))
    print(len(data_no_flaw))
    data_flaw = set()
    data_no_flaw = set()
    # data_un = set()
# remember this is write

with open('sard_multi_replace_tokens_Tokens.pkl', 'wb') as f:
    pickle.dump(data_ls, f)


print(no_flaws)

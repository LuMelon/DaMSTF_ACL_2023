import os

def parseItems(items):
    tgt_dirs = [os.path.join(items[0], dname) for dname in items[1] if dname == '__pycache__' or dname == '.ipynb_checkpoints' ]
    return tgt_dirs


def generateIgnoreFiles(dirName):
    fnames = [os.path.join(item[0], fname) for item in os.walk(dirName) for fname in item[2] if fname.endswith('.pkl')]
    with open('.gitignore', 'a') as fw:
        fw.write("\n".join(fnames))
# tgts = [dname for fname in os.walk('.') for dname in parse_items(fname)]
# [os.system(f'rm -rf {d_name}') for d_name in tgts]


if __name__ == '__main__':
    pass
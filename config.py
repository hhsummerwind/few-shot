import os


PATH = os.path.dirname(os.path.realpath(__file__))

# DATA_PATH = '/data/datasets/classify/omniglot'
DATA_PATH = '/data/datasets/tianji/qmsb/202109事前签名比对/new/few_shot'
EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')

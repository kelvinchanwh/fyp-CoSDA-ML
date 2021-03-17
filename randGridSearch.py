import logging
import numpy as np
import random
import torch
import util.tool
import pandas as pd
from util.configue import Configure
import os.path
import time

from bayes_opt import BayesianOptimization

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# PATH = "./exp"
PATH = "/content/drive/MyDrive/CoSDA-ML/"

# Bounded region of parameter space
pbounds = {'ratio': (0, 1.0), 'cross': (0, 1.0), 'invratio':(0, 1.0)}

filename = "SC2_bert.json"
stop_key = 'eval_test_de_joint_goal'

# # DST
# 'eval_test_de_joint_goal'
# 'eval_test_de_turn_area' 
# 'eval_test_de_turn_food' 
# 'eval_test_de_turn_inform'
# 'eval_test_de_turn_price'
# 'eval_test_de_turn_request'
# 'eval_test_it_joint_goal'
# 'eval_test_it_turn_area'
# 'eval_test_it_turn_food'
# 'eval_test_it_turn_inform'
# 'eval_test_it_turn_price'
# 'eval_test_it_turn_request'

# # MLDoc
# 'eval_MLDoc/chinese.test_accuracy'
# 'eval_MLDoc/english.test_accuracy'
# 'eval_MLDoc/french.test_accuracy'
# 'eval_MLDoc/german.test_accuracy'
# 'eval_MLDoc/italian.test_accuracy'
# 'eval_MLDoc/japanese.test_accuracy'
# 'eval_MLDoc/russian.test_accuracy'
# 'eval_MLDoc/spanish.test_accuracy'

# # SC2 / SC4
# 'eval_MIXSC/ca/opener_sents_f1'
# 'eval_MIXSC/es/opener_sents_f1'
# 'eval_MIXSC/eu/opener_sents_f1'

# # XTDS
# 'eval_XTDS/es/test-es.conllu_intent_accuracy'
# 'eval_XTDS/es/test-es.conllu_slot_f1'
# 'eval_XTDS/th/test-th_TH.conllu_intent_accuracy'
# 'eval_XTDS/th/test-th_TH.conllu_slot_f1'

def start(ratio, cross, invratio):

    logging.basicConfig(level = logging.INFO)

    args = Configure.Get()

    np.random.seed(args.train.seed)
    random.seed(args.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train.seed)
        torch.cuda.manual_seed(args.train.seed)

    torch.manual_seed(args.train.seed)
    torch.random.manual_seed(args.train.seed)

    Model, DatasetTool = util.tool.load_module(args.model.name, args.dataset.tool)

    args.train.stop_key = stop_key

    #Insert variables to Grid Search
    args.train.ratio = ratio
    args.train.cross = cross
    args.train.invratio = invratio

    inputs = DatasetTool.get(args)

    model = Model(args, DatasetTool, inputs)
    if args.train.tpu:
        torch.set_default_tensor_type('torch.FloatTensor')

    toReturn = xmp.spawn(model.start, args=(inputs,), nprocs=8, start_method='fork')

    return toReturn

# if __name__ == "__main__":
#     start()

if os.path.isfile(PATH + filename):
    #Load Previous JSON Logs
    optimizer = BayesianOptimization(
        f=start,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    load_logs(optimizer, logs=[PATH + filename])
    print ("Optimizer has loaded {} points from previous JSON".format(len(optimizer.space)))

else:
    #Do new logs
    optimizer = BayesianOptimization(
        f=start,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

logger = JSONLogger(path=PATH+filename, reset=False)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=5,
    n_iter=25,
)

# print(optimizer.max)
# for i, res in enumerate(optimizer.res):
#     print("Iteration {}: \n\t{}".format(i, res))
import logging
import numpy as np
import random
import torch
import util.tool
import pandas as pd
from util.configue import Configure
import os.path
import time

# from bayes_opt import BayesianOptimization

# from bayes_opt.logger import JSONLogger
# from bayes_opt.event import Events
# from bayes_opt.util import load_logs

#SETTINGS

# PATH = "./exp"
PATH = "/content/drive/MyDrive/CoSDA-ML/"

# Bounded region of parameter space
pbounds = {'ratio': (0, 1.0), 'cross': (0, 1.0), 'invratio':(0, 1.0)}

#Output Filename
filename = "SC2_bert.json"

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

    # args.train.stop_key = stop_key

    #Insert variables to Grid Search
    args.train.ratio = ratio
    args.train.cross = cross
    args.train.invratio = invratio

    inputs = DatasetTool.get(args)

    model = Model(args, DatasetTool, inputs)
    if args.train.gpu:
        model.cuda()

    return model.start(inputs)


# if os.path.isfile(PATH + filename):
#     #Load Previous JSON Logs
#     optimizer = BayesianOptimization(
#         f=start,
#         pbounds=pbounds,
#         target_key=target,
#         verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#         random_state=1,
#     )
#     load_logs(optimizer, logs=[PATH + filename])
#     print ("Optimizer has loaded {} points from previous JSON".format(len(optimizer.space)))

# else:
#     #Do new logs
#     optimizer = BayesianOptimization(
#         f=start,
#         pbounds=pbounds,
#         target_key = target,
#         verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#         random_state=1,
#     )

# logger = JSONLogger(path=PATH+filename, reset=False)
# optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# optimizer.maximize(
#     init_points=init_points,
#     n_iter=n_iter,
# )

values = [x*0.1 for x in range (0,11)]
ratio = 1.0
cross = 1.0
invratio = 1.0

for invratio in values:
    with open(PATH+filename, 'w+') as write_file:
        out = start(ratio, cross, invratio)
        write_file.write(out)
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

# PATH = "./exp"
PATH = "/content/drive/MyDrive/CoSDA-ML/"

# Bounded region of parameter space
pbounds = {'ratio': (0, 1.0), 'cross': (0, 1.0), 'invratio':(0, 1.0)}

filename = "SC2_bert.json"
timestr = time.strftime("%Y%m%d-%H%M%S")

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

    #Insert variables to Grid Search
    args.train.ratio = ratio
    args.train.cross = cross
    args.train.invratio = invratio

    inputs = DatasetTool.get(args)

    model = Model(args, DatasetTool, inputs)
    if args.train.gpu:
        model.cuda()

    return model.start(inputs)

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
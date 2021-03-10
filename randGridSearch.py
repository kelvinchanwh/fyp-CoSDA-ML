import logging
import numpy as np
import random
import torch
import util.tool
import pandas as pd
from util.configue import Configure

from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'x': (0, 1.0), 'y': (0, 1.0), 'z':(0, 1.0)}

def start(x, y, z):

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
    args.train.ratio = x
    args.train.cross = y
    args.train.invratio = z

    inputs = DatasetTool.get(args)

    model = Model(args, DatasetTool, inputs)
    if args.train.gpu:
        model.cuda()

    return model.start(inputs)

# if __name__ == "__main__":
#     start()

optimizer = BayesianOptimization(
    f=start,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)
    
optimizer.maximize(
    init_points=5,
    n_iter=25,
)

print(optimizer.max)
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
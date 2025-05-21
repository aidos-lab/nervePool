import pickle

import numpy as np
import torch

#######################################################################
### Load fixtures
#######################################################################


with open("./test/fixtures/simplices.pkl", "rb") as f:
    simplices = pickle.load(f)

with open("./test/fixtures/boundaries.pkl", "rb") as f:
    boundaries = pickle.load(f)


with open("./test/fixtures/adjacencies.pkl", "rb") as f:
    adjacencies = pickle.load(f)


breakpoint()

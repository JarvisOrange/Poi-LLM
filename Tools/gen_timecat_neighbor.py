import torch
from torch import einsum, nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm

from datetime import datetime
import time
from geopy.distance import geodesic
import itertools
import math
import random
from random import choice


dataset_path_dict = {
    'NY':'./Dataset/Foursquare_NY/nyc.geo',
    'SG':'./Dataset/Foursquare_SG/singapore.geo',
    'TKY':'./Dataset/Foursquare_TKY/tky.geo',
}

def gen_timecat_neighbor():
    return 0
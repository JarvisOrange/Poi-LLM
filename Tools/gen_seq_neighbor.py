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
    'NY':'./Dataset/Foursquare_NY/nyc.poitraj',
    'SG':'./Dataset/Foursquare_SG/singapore.poitraj',
    'TKY':'./Dataset/Foursquare_TKY/tky.poitraj',
}
        

def gen_seq_neighbor(dataset_name, , save_path=None):
    data_path = dataset_path_dict[dataset_name]

    poi_df = pd.read_csv(data_path, sep=',', header=None)

    columns_standard = ["entity_id","location","time","type","dyna_id"]
    if dataset_name != 'TKY':
        poi_df.columns = columns_standard
    else:
        poi_df.columns = ["dyna_id","type","time","entity_id","location"]
        poi_df = poi_df.loc[:, columns_standard]
    

    location_group = poi_df.groupby("location")

    seq_sample = {}
    

    for l_g in tqdm(location_group):
        locations = pd.DataFrame(l_g[1])
        length = len(locations)
        
        for i in range(len(locations)):
            poi_id = l_g['location'][i]
            time = l_g['time'][i]

        for i in range(len(poi_seq)):
            seq_sample[poi_seq[i]] = poi_seq[max(0,i - window): min(length,i+window+1)]
        

        
    


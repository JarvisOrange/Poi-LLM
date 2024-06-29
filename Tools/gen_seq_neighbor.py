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

monthdict={
    'Jan':1,
    'Feb':2,
    'Mar':3,
    'Apr':4,
    'May':5,
    'Jun':6,
    'Jul':7,
    'Aug':8,
    'Sep':9,
    'Oct':10,
    'Nov':11,
    'Dec':12
}

def get_day(time):
    if '-' in time:
        date = time.split("T")[0]
        time = datetime.strptime(date,"%Y-%m-%d")
        return (int(time.year), int(time.month), int(time.day))
    else:
        temp = time.split(" ")
        date = monthdict[temp[1]] * 100 + int(temp[2])
        year = int(temp[-1])
        month = int(monthdict[temp[1]])
        day = int(temp[2])
        return (year, month, day)
        

def gen_seq_neighbor(dataset_name, window=2,  save_path=None):
    data_path = dataset_path_dict[dataset_name]

    poi_df = pd.read_csv(data_path, sep=',', header=0)

    columns_standard = ["entity_id","location","time","type","dyna_id"]
    if dataset_name != 'TKY':
        poi_df.columns = columns_standard
    else:
        poi_df.columns = ["dyna_id","type","time","entity_id","location"]
        poi_df = poi_df.loc[:, columns_standard]
    

    traj_group = poi_df.groupby("entity_id")

    seq_sample = {}
    
    for l_g in tqdm(traj_group):
        traj = pd.DataFrame(l_g[1])
        length = len(traj)
        
        traj['day'] = traj['time'].apply(get_day)
        
        for i in range(length):
            poi_id = traj.iloc[i, 1] #location id
            poi_day = traj.iloc[i,5] #day time
            temp= traj.iloc[max(0,i - window): min(length,i + window + 1), :]
            temp = temp[(temp['day'] == poi_day) & (temp['location'] != poi_id)]
            if len(temp) != 0:
                if poi_id not in seq_sample:
                    seq_sample[poi_id] = []
                seq_sample[poi_id] = seq_sample[poi_id]+list((temp)['location'])
        
    for k in seq_sample.keys():
        seq_sample[k] = set(seq_sample[k])

        
    

gen_seq_neighbor('NY')
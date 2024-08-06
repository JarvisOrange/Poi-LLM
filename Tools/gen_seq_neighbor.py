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
import os

# dataset_path_dict = {
#     'NY':'./Dataset/NY/ny_ttraj.csv',
#     'SG':'./Dataset/SG/sg_traj.csv',
#     'TKY':'./Dataset/TKY/tky_traj.csv',
# }

dataset_path_dict = {
    'NY':'./Washed/common/ny_train_traj.csv',
    'SG':'./Dataset/SG/sg_traj.csv',
    'TKY':'./Dataset/TKY/tky_traj.csv',
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

# def get_day(time):
#     if '-' in time:
#         date = time.split("T")[0]
#         time = datetime.strptime(date,"%Y-%m-%d")
#         return (int(time.year), int(time.month), int(time.day))
#     else:
#         temp = time.split(" ")
#         date = monthdict[temp[1]] * 100 + int(temp[2])
#         year = int(temp[-1])
#         month = int(monthdict[temp[1]])
#         day = int(temp[2])
#         return (year, month, day)
    
def get_day(time):
    date = time.split(" ")[0]
    time = datetime.strptime(date,"%Y-%m-%d")
    return (int(time.year), int(time.month), int(time.day))
   
        

def gen_seq_neighbor(dataset_name, window=2,   save_path="./Washed_ContrastDataset/"):
    data_path = dataset_path_dict[dataset_name]

    poi_df = pd.read_csv(data_path, sep=',', header=0)

    # columns_standard = ["index","entity_id","location","time","type","dyna_id"]
    columns_standard = ["index","entity_id","location","time","dyna_id"]
    if dataset_name != 'TKY':
        poi_df.columns = columns_standard
    else:
        # poi_df.columns = ["index","dyna_id","type","time","entity_id","location"]
        poi_df.columns = ["index","dyna_id","time","entity_id","location"]
        poi_df = poi_df.loc[:, columns_standard]

    

    traj_group = poi_df.groupby("entity_id")

    seq_sample = {}
    
    for l_g in tqdm(traj_group):
        traj = pd.DataFrame(l_g[1])
        length = len(traj)
        
        traj['day'] = traj['time'].apply(get_day)

        
        
        for i in range(length):
            poi_id = traj.iloc[i, 2] #location id
            # poi_day = traj.iloc[i, 6] #day time
            poi_day = traj.iloc[i, 3] #day time
            temp= traj.iloc[max(0,i - window): min(length,i + window + 1), :]
            temp = temp[(temp['day'] == poi_day) & (temp['location'] != poi_id)]
            if len(temp) != 0:
                if poi_id not in seq_sample:
                    seq_sample[poi_id] = []
                seq_sample[poi_id] = seq_sample[poi_id]+list((temp)['location'])

    x= 0 # 10.545360110803324
    seq_neighbor_list = []
    for k in seq_sample.keys():
        temp = list(set(seq_sample[k]))
        seq_sample[k] = temp
        seq_neighbor_list.append([k,temp])

    seq_neighbor_df = pd.DataFrame(seq_neighbor_list, columns=['geo_id','seq_positive'])

    save_path = save_path +'/'+ dataset_name +'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    # name =  dataset_name + "_seq_positive_train.csv"
    name =  dataset_name + "_seq_positive_train.csv"

    seq_neighbor_df.to_csv(save_path + name, sep=',', index=False, header=True)


    
for dataset in ['NY']:   
    gen_seq_neighbor(dataset)  

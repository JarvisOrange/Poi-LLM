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

def gen_timecat_neighbor(dataset_name, save_path=None):
    data_path = dataset_path_dict[dataset_name]

    columns_standard = ["poi_id","coordinates","category_name"]
    if dataset_name == 'TKY':
        columns_read = ['geo_id','coordinates','venue_category_name']
        poi_df = pd.read_csv(data_path, sep=',', header=0, usecols=['geo_id','coordinates','venue_category_name'])
    else:
        columns_read = ['type','coordinates','poi_type','poi_id']
        poi_df = pd.read_csv(data_path, sep=',', header=0, usecols=columns_read)
        poi_df = poi_df[poi_df['type']=='Point']
        poi_df = poi_df.drop(['type'], axis=1)
        poi_df = poi_df.loc[:,['poi_id','coordinates','poi_type']]
    poi_df.columns = columns_standard 
    poi_df['lon'] = poi_df['coordinates'].apply(lambda x: eval(x)[0])
    poi_df['lat'] = poi_df['coordinates'].apply(lambda x: eval(x)[1])
    
    ###    
    poi_time_class = pd.read_csv('')
    ###

    time_cat_neighbor_dict = {}
    for _, row in tqdm(poi_df.iterrows()):
        poi_id = row['poi_id']
        poi_cat, time_class = row['poi_type'],  row['time_class']
        temp = poi_df[(poi_df['poi_type'] == poi_cat) 
                      & (temp['time_class'] == time_class) 
                      & (poi_df['poi_id'] == poi_id)]
        time_cat_neighbor_dict['poi_id'] = list(temp)

        
    

gen_seq_neighbor('NY')
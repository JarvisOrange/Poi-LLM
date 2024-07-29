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

dataset_path_dict = {
    'NY':'./Dataset/Foursquare_NY/ny.geo',
    'SG':'./Dataset/Foursquare_SG/sg.geo',
    'TKY':'./Dataset/Foursquare_TKY/tky.geo',
}

time_dataset_path_dict = {
    'NY':'./Feature/NY/poi_NY_time.csv',
    'SG':'./Feature/SG/poi_SG_time.csv',
    'TKY':'./Feature/TKY/poi_TKY_time.csv',
}


def gen_timecat_neighbor(dataset_name, save_path="./ContrastDataset/"):
    data_path = dataset_path_dict[dataset_name]

    columns_standard = ["geo_id","coordinates","category"]
    if dataset_name == 'TKY':
        columns_read = ['geo_id','coordinates','venue_category_name']
        poi_df = pd.read_csv(data_path, sep=',', header=0, usecols=['geo_id','coordinates','venue_category_name'])
    else:
        columns_read = ['geo_id','type','coordinates','poi_type']
        poi_df = pd.read_csv(data_path, sep=',', header=0, usecols=columns_read)
        poi_df = poi_df[poi_df['type']=='Point']
        poi_df = poi_df.drop(['type'], axis=1)
        poi_df = poi_df.loc[:,['geo_id','coordinates','poi_type']]
    poi_df.columns = columns_standard 
    poi_df['lon'] = poi_df['coordinates'].apply(lambda x: eval(x)[0])
    poi_df['lat'] = poi_df['coordinates'].apply(lambda x: eval(x)[1])

    first = poi_df.iloc[0,0]
    poi_df['geo_id'] = poi_df['geo_id'].apply(lambda x: x - first)
    

    time_data_path = time_dataset_path_dict[dataset_name]
    poi_time_df = pd.read_csv(time_data_path, sep=',', header=0)

    first = poi_df.iloc[0,0]
   
    poi_df['geo_id'] = poi_df['geo_id'].apply(lambda x: x - first)
    poi_df = pd.merge(poi_df, poi_time_df, how='inner', on='geo_id')

    
    x= 0 # 269.7588670826732
    time_cat_neighbor_list = []
    for _, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0]):
        poi_id = row['geo_id']
        poi_cat, day_feature, hour_feature = row['category'],  row['day_feature'], row['hour_feature']
        temp = poi_df[(poi_df['category'] == poi_cat) &
                      (poi_df['day_feature'] == day_feature) &
                      (poi_df['hour_feature'] == hour_feature) &
                      (poi_df['geo_id'] != poi_id)]
        temp = list(temp['geo_id'])
        time_cat_neighbor_list.append([poi_id,temp])

        x+=len(temp)

    print(x/poi_df.shape[0])
    
    time_cat_neighbor_df = pd.DataFrame(time_cat_neighbor_list, columns=['geo_id','time_cat_positive'])
    save_path = save_path +'/'+ dataset_name +'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    name =  dataset_name + "_time_cat_positive.csv"
    time_cat_neighbor_df.to_csv(save_path + name, sep=',', index=False, header=True)


        
    

gen_timecat_neighbor('TKY')
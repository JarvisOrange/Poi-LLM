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

def cal_degree(lon, lat, min_distance=0.5):
    r = 6371
    dlon =  2 * math.asin(math.sin(min_distance/(2*r))/math.cos(lat * math.pi/180))
    dlon = dlon * 180 / math.pi
    dlat = min_distance / r * 180 / math.pi;	
    minlat = lat- dlat
    maxlat = lat + dlat
    minlon = lon - dlon
    maxlon = lon + dlon
    return (minlon, maxlon, minlat, maxlat)


def get_geo_cat_neighbor(poi_list, poi_id, cat, minlon, maxlon, minlat, maxlat):
    poi_geo_cat_neighbor = poi_list[(poi_list['geo_id'] !=poi_id) &
                               (poi_list['lon'] > minlon) & 
                               (poi_list['lon'] < maxlon) & 
                               (poi_list['lat'] > minlat) & 
                               (poi_list['lat'] < maxlat) &
                               (poi_list['category'] == cat)
                               ]
    return poi_geo_cat_neighbor


def gen_poi_geoneighbor(dataset_name,  save_path="./ContrastDataset/"):
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



    geoneighbor_list = []
    # x = 0  #平均23.42181672045914
    for _, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0]):
        poi_id = row['geo_id']
        cat= row['category']
        lon, lat = row['lon'],  row['lat']
        minlon, maxlon, minlat, maxlat = cal_degree(lon, lat)
        poi_geo_cat_neighbor = get_geo_cat_neighbor(poi_df, poi_id, cat, minlon, maxlon, minlat, maxlat)
        temp =  list(poi_geo_cat_neighbor['geo_id'])
        geoneighbor_list.append([poi_id, temp])
        
    geoneighbor_df = pd.DataFrame(geoneighbor_list, columns=['geo_id','geo_positive'])

    save_path = save_path +'/'+ dataset_name +'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    name =  dataset_name + "_geo_positive.csv"
    geoneighbor_df.to_csv(save_path + name, sep=',', index=False, header=True)


    
        
      
        
    
gen_poi_geoneighbor('TKY')

    
        
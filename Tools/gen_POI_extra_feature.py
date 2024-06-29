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
import time
import os



dataset_traj_path_dict = {
    'NY':'./Dataset/Foursquare_NY/nyc.poitraj',
    'SG':'./Dataset/Foursquare_SG/singapore.poitraj',
    'TKY':'./Dataset/Foursquare_TKY/tky.poitraj',
}


dataset_geo_path_dict = {
    'NY':'./Dataset/Foursquare_NY/nyc.geo',
    'SG':'./Dataset/Foursquare_SG/singapore.geo',
    'TKY':'./Dataset/Foursquare_TKY/tky.geo',
}

### gen poi visit-time feature
def get_weekday_weekend(time):
        if '-' in time:
            time = datetime.strptime(time,"%Y-%m-%d")
            if time.weekday() <= 4:
                return "weekday"
            else:
                return "weekend"
        else:
            Work = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
            if time in Work:
                return "weekday"
            else:
                return "weekend"
        

def get_daytime(time):
    '''
    time class 0: 7:00:00-10:59:59
    time class 1: 11:00:00-13:59:59
    time class 2: 14:00:00-16:59:59
    time class 3: 17:00:00-20:59:59
    time class 4: 21:00:00-06:59:59
    '''

    time = datetime.strptime(time,"%H:%M:%S")
    if time.hour >= 6 and time.hour < 11:
        return 0
    elif time.hour >= 11 and time.hour < 14:
        return 1
    elif time.hour >= 14 and time.hour < 17:
        return 2
    elif time.hour >= 17 and time.hour < 21:
        return 3
    else:
        return 4

def gen_poi_time_feature(dataset_name,  save_path=None):
        
    data_path = dataset_traj_path_dict[dataset_name]
    poi_df = pd.read_csv(data_path, sep=',', header=0)

    columns_standard = ["entity_id","location","time","type","dyna_id"]
    if dataset_name != 'TKY':
        poi_df.columns = columns_standard
    else:
        poi_df.columns = ["dyna_id","type","time","entity_id","location"]
        poi_df = poi_df.loc[:, columns_standard]


    poi_time_feature_result = []

    location_group = poi_df.groupby("location")
    

    for l_g in tqdm(location_group):
        locations = pd.DataFrame(l_g[1])
        if dataset_name == "TKY":
            locations['date'] = locations['time'].apply(lambda x: x.split('T')[0])
            locations['daytime'] = locations['time'].apply(lambda x: x.split('T')[1][:-1])

        else:
            locations['date'] = locations['time'].apply(lambda x: x.split(' ')[0])
            locations['daytime'] = locations['time'].apply(lambda x: x.split(' ')[3])
        
        locations['date'] = locations['date'].apply(get_weekday_weekend)
        locations['daytime'] = locations['daytime'].apply(get_daytime)

        poi_id = locations.iloc[0, 1] # second column is 'location'
       
        total = len(l_g[1])

        count_temp1 = {"weekday":0, "weekend":0}
        count_temp2 = {0:0, 1:0, 2:0, 3:0, 4:0}

        temp_dict = dict(locations['date'].value_counts())
        for k in temp_dict.keys():
            count_temp1[k] += count_temp1[k] + temp_dict[k]

        temp_dict = dict(locations['daytime'].value_counts())
        for k in temp_dict.keys():
            count_temp2[k] += count_temp2[k] + temp_dict[k]

        day_feature = ("weekday"if count_temp1['weekday'] > count_temp1['weekend'] else "weekend")
        hour_feature = max(count_temp2, key=lambda x: count_temp2[x])


        poi_time_feature_result.append([poi_id, total, day_feature, hour_feature])

    poi_time_feature_df = pd.DataFrame(poi_time_feature_result, columns=['geo_id','total_visit_time','day_feature','hour_feature'])

    if not os.path.exists(save_path):
            os.makedirs(save_path)
    name = 'poi' + "_" + dataset_name + "_time.csv"
    save_path_name = save_path  + name
    poi_time_feature_df.to_csv(save_path_name, sep=',', index=False, header=True)
      



        
### gen poi poi-nearby feature
def cal_degree(lon, lat, min_distance=0.05):
        r = 6371
        dlon =  2 * math.asin(math.sin(min_distance/(2*r))/math.cos(lat * math.pi/180))
        dlon = dlon * 180 / math.pi
        dlat = min_distance / r * 180 / math.pi;	
        minlat = lat - dlat
        maxlat = lat + dlat
        minlon = lon - dlon
        maxlon = lon + dlon
        return (minlon, maxlon, minlat, maxlat)


def get_geoneighbor(poi_list, poi_id, minlon, maxlon, minlat, maxlat):
    poi_geoneighbor = poi_list[(poi_list['poi_id'] !=poi_id) &
                            (poi_list['lon'] > minlon) & 
                            (poi_list['lon'] < maxlon) & 
                            (poi_list['lat'] > minlat) & 
                            (poi_list['lat'] < maxlat)]
    return poi_geoneighbor

def cal_poi_category(poi_cat_df):
    cat_dict = poi_cat_df.value_counts().to_dict()
    if len(cat_dict) == 0:
        return " "
    if len(cat_dict) < 3:
        return  ",".join(list(cat_dict.keys()))
    else:
        sorted_dict = sorted(cat_dict.items(), key=lambda x: x[1], reverse = True)
        return  ",".join(list(cat_dict.keys())[0:3])


def gen_poi_category_feature(dataset_name,  save_path=None):

    data_path = dataset_geo_path_dict[dataset_name]

    columns_standard = ["poi_id","coordinates","category_name"]
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
    
    
    
    geoneighbor_dict = {}
    poi_category_feature_result = []
    for _, row in tqdm(poi_df.iterrows()):
        poi_id = row['poi_id']
        cat = row['category_name']
        lon, lat = row['lon'],  row['lat']
        minlon, maxlon, minlat, maxlat = cal_degree(lon, lat)
        poi_geoneighbor = get_geoneighbor(poi_df, poi_id, minlon, maxlon, minlat, maxlat)
        geoneighbor_dict['poi_id'] =  poi_geoneighbor['poi_id']
        temp = cal_poi_category(poi_geoneighbor['category_name'])

        poi_category_feature_result.append([poi_id, cat, temp])


    poi_category_feature_df = pd.DataFrame(poi_category_feature_result, columns=['geo_id','category','category_nearby'])

    if not os.path.exists(save_path):
            os.makedirs(save_path)
    name = 'poi' + "_" + dataset_name + "_cat_nearby.csv"
    save_path_name = save_path  + name
    poi_category_feature_df.to_csv(save_path_name, sep=',', index=False, header=True)

    



dataset = 'NY'

save_path = "./Feature/" + dataset + "/"
gen_poi_category_feature(dataset, save_path= save_path)



        

       






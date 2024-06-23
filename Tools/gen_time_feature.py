import torch
from torch import einsum, nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm

from datetime import datetime
import time


dataset_path_dict = {
    'NY':'./Dataset/Foursquare_NY/nyc.poitraj',
    'SG':'./Dataset/Foursquare_SG/singapore.poitraj',
    'TKY':'./Dataset/Foursquare_TKY/tky.poitraj',
}

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


def gen_poi_time(dataset_name,  save_path=None):
    data_path = dataset_path_dict[dataset_name]
    poi_df = pd.read_csv(data_path, sep=',', header=None)

    columns_standard = ["entity_id","location","time","type","dyna_id"]
    if dataset_name != 'TKY':
        poi_df.columns = columns_standard
    else:
        poi_df.columns = ["dyna_id","type","time","entity_id","location"]
        poi_df = poi_df.loc[:, columns_standard]
    

    poi_time_df = pd.DataFrame(data=None, columns = columns_standard)

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

        total = len(l_g[1])
        count_temp1 = {"weekday":0, "weekend":0}
        count_temp2 = {0:0, 1:0, 2:0, 3:0, 4:0}
        temp_dict = dict(locations['date'].value_counts())
        for k in temp_dict.keys():
            count_temp1[k] += count_temp1[k] + temp_dict[k]

        temp_dict = dict(locations['daytime'].value_counts())
        for k in temp_dict.keys():
            count_temp2[k] += count_temp2[k] + temp_dict[k]

        locations['freq'] = total

        locations['weekday'] = count_temp1['weekday'] 
        locations['weekday'] = count_temp1['weekend'] 

        locations['time_class_0'] = count_temp2[0] 
        locations['time_class_1'] = count_temp2[1]
        locations['time_class_2'] = count_temp2[2]
        locations['time_class_3'] = count_temp2[3]
        locations['time_class_4'] = count_temp2[4]

        
        
        


gen_poi_time('TKY')

        

       






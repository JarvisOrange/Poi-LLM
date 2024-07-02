import math
import os
import time
import json
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

time_dict = {
    0: '6:00:00-8:59:59',
    1: '9:00:00-10:59:59',
    2: '11:00:00-12:59:59',
    3: '13:00:00-16:59:59',
    4: '17:00:00-18:59:59',
    5: '19:00:00-23:59:59',
    6: '00:00:00-5:59:59'
}



def create_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="time",
        choices=['address', 'time','cat_nearby'],
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="NY",
        choices=["NY", "SG", "TKY"],
        help="which dataset",
    )

    args = parser.parse_args()

    return args



def main(): 
    args = create_args()
    
    prompt_type = args.prompt_type

    dataset_name = args.dataset

    poi_data_path = "./Dataset/Foursquare_" + dataset_name+ '/' + dataset_name.lower() + '.geo'
    feature_data_path = "./Feature/" + "" + dataset_name +"/"+ "poi_" + dataset_name + "_" + prompt_type + '.csv'

    columns_standard = ["geo_id", "coordinates", "category"]

    if dataset_name == 'TKY':
        columns_read = ['geo_id','coordinates','venue_category_name']
        poi_df = pd.read_csv(poi_data_path, sep=',', header=0, usecols=columns_read)
    else:
        columns_read = ['geo_id','type','coordinates','poi_type']
        poi_df = pd.read_csv(poi_data_path, sep=',', header=0, usecols=columns_read)
        poi_df = poi_df[poi_df['type']=='Point']

        first = poi_df.iloc[0,0]
   
        poi_df = poi_df.drop(['type'], axis=1)
        poi_df = poi_df.loc[:,['geo_id','coordinates','poi_type']]

        poi_df['geo_id'] = poi_df['geo_id'].apply(lambda x: x - first)

    poi_df.columns = columns_standard 


    poi_feature_df = pd.read_csv(feature_data_path, sep=',', header=0)

    poi_df = pd.merge(poi_df,poi_feature_df, how='inner', on='geo_id')
    

    prompt_result = []
    prompt_base = "The poi is located in"

    if prompt_type == 'address':
        for _, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0]):
            # postcode,housenumber,street,city,country
        
    elif prompt_type == 'time':
        for _, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0]):
            day = row['day_feature'] #day feature
            hour = row['hour_feature']

            prompt_time = "POI's Time Information: And People usually visit this place in the " + day +'.'
            
            
    elif prompt_type == 'category':
        for _, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0]):
            category = row['category']
            category_nearby = row['category_nearby']

            prompt_category = "POI's Category Information: And the poi is a" + category + ','
            prompt_category += "There are " + cat_nearby + " near this POI."

            
            




if __name__ == "__main__":
    main()
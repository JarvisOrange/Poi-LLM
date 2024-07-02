import math
import os
import time
import json
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

time_dict = {
    0: 'between 6 am and 9 am',
    1: 'between 9 am and 11 am',
    2: 'between 11 am and 1 pm',
    3: 'between 1 pm and 5 pm',
    4: 'between 5 pm and 7 pm',
    5: 'between 7 pm and 12 pm',
    6: 'between 12 pm and 6 am the next day'
}

dataset_city_dict = {
    'NY': 'New York',
    'SG': 'Singapore',
    'TKY': 'Tokyo',
}


def save_prompt(prompt_result, data_path):
    df = pd.DataFrame(prompt_result)
    df.insert(0, 'geo_id', range(len(df)), allow_duplicates=False)
    df.columns=['geo_id','prompt']
    
    df.to_csv(data_path, index=False, header=True)

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

    columns_standard = ["geo_id", "coordinates"]

    if dataset_name == 'TKY':
        columns_read = ['geo_id','coordinates']
        poi_df = pd.read_csv(poi_data_path, sep=',', header=0, usecols=columns_read)
    else:
        columns_read = ['geo_id','type','coordinates']
        poi_df = pd.read_csv(poi_data_path, sep=',', header=0, usecols=columns_read)
        poi_df = poi_df[poi_df['type']=='Point']

        first = poi_df.iloc[0,0]
   
        poi_df = poi_df.drop(['type'], axis=1)
        poi_df = poi_df.loc[:,['geo_id','coordinates']]

        poi_df['geo_id'] = poi_df['geo_id'].apply(lambda x: x - first)

    poi_df.columns = columns_standard 


    poi_feature_df = pd.read_csv(feature_data_path, sep=',', header=0)

    poi_df = pd.merge(poi_df,poi_feature_df, how='inner', on='geo_id')

    prompt_result = []

    prompt_base = "The latitude and longitude of the POI are "

    if prompt_type == 'address':
        for _, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0]):
            prompt = ""
            temp = eval(row['coordinates'])
            lat, lon = temp[0], temp[1]
            if lat < 0:
                lat = str(-lat) +" South"
            else:
                lat = str(lat) +" North"
            if lon < 0:
                lon = str(-lon) +" West"
            else:
                lon = str(lon) +" East"

            prompt += prompt_base + lon +" and " + lat + '.'
            
            postcode = row['postcode']
            housenumber = row['housenumber']
            street = row['street']
            country = row['country']

            # prompt_address = 

        
    elif prompt_type == 'time':
        for _, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0]):
            prompt = ""
            temp = eval(row['coordinates'])
            lat, lon = temp[0], temp[1]
            if lat < 0:
                lat = str(-lat) +" South"
            else:
                lat = str(lat) +" North"
            if lon < 0:
                lon = str(-lon) +" West"
            else:
                lon = str(lon) +" East"

            prompt += prompt_base + lon +" and " + lat + '.'

            day = row['day_feature'] 
            hour = row['hour_feature']

            prompt+= "\n"+"Time Information: people usually visit this POI " + time_dict[hour]+", "
            prompt+= "and usually come to this POI on " + day +'s.'

            prompt_result.append(prompt)

            
            
            
            
    elif prompt_type == 'cat_nearby':
        for _, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0]):
            prompt = ""
            temp = eval(row['coordinates'])
            lat, lon = temp[0], temp[1]
            if lat < 0:
                lat = str(-lat) +" South"
            else:
                lat = str(lat) +" North"
            if lon < 0:
                lon = str(-lon) +" West"
            else:
                lon = str(lon) +" East"

            prompt += prompt_base + lon +" and " + lat + '.'

            category = row['category']
            category_nearby = row['category_nearby']

            prompt += "\n"+"Category Information: the POI is a " + category + "."

            if category_nearby != " ":   
                prompt += "And there are " + category_nearby + " near this POI."

            prompt += "\n"+"Where is the POI in " + dataset_city_dict[dataset_name] +" ?"

            prompt_result.append(prompt)


    save_data_path = "./Prompt/" + "" + dataset_name +"/"+ "prompt_" + dataset_name + "_" + prompt_type + '.csv'
    save_prompt(prompt_result, save_data_path)
            
            
            




if __name__ == "__main__":
    main()
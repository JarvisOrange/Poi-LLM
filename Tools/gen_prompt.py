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

    name_data_path = "./Feature/" + "" + dataset_name +"/"+ "poi_" + dataset_name + "_" + 'address' + '.csv'

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

    poi_feature_df = pd.read_csv(feature_data_path, sep=',', header=0, dtype={'osm_calculated_postcode':str})

    

    poi_df = pd.merge(poi_df, poi_feature_df, how='outer', on='geo_id')

    if prompt_type != 'address':

        name_df = pd.read_csv(name_data_path, sep=',', header=0, dtype={'osm_calculated_postcode':str})

        poi_df = pd.merge(poi_df, name_df, how='outer', on='geo_id')

    poi_df=poi_df.fillna("")
   
    prompt_result = []

    prompt_base = "The latitude and longitude of the POI are "

    if prompt_type == 'address':

        for _, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0]):
            prompt = ""

            name = ""
            name_info = eval(row['osm_names'])
            if "name:en" in name_info:
                name = name_info['name:en']
            elif "name" in name_info:
                name = name_info['name']
            if name !='':
                prompt +=  "The name of this POI is " + name + ". "

            temp = eval(row['coordinates'])
            lon, lat = temp[0], temp[1]
            if lat < 0:
                lat = str(-lat) +" South"
            else:
                lat = str(lat) +" North"
            if lon < 0:
                lon = str(-lon) +" West"
            else:
                lon = str(lon) +" East"

            prompt += prompt_base + lat +" and " + lon + '.'

            prompt += "\n" + "Address Information: "

            

            

            housenumber = row['housenumber']
            street = row['street']
            if street !='' and housenumber !='':
                prompt +=  " The POI is located at " + housenumber + " " +street + "."
            
            postcode = str(row['osm_calculated_postcode'])
            if postcode !='':
                prompt +=  " The postcode of the POI is " + postcode + "."

            prompt += "\n"+"Question: Where is the POI in " + dataset_city_dict[dataset_name] +"?"
            prompt_result.append(prompt)
        
    elif prompt_type == 'time':
        for _, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0]):
            prompt = ""

            name = ""
            name_info = eval(row['osm_names'])
            if "name:en" in name_info:
                name = name_info['name:en']
            elif "name" in name_info:
                name = name_info['name']
            if name !='':
                prompt +=  "The name of the POI is " + name + ". "


            temp = eval(row['coordinates'])
            lon, lat = temp[0], temp[1]
            if lat < 0:
                lat = str(-lat) +" South"
            else:
                lat = str(lat) +" North"
            if lon < 0:
                lon = str(-lon) +" West"
            else:
                lon = str(lon) +" East"

            prompt += prompt_base + lat +" and " + lon + '.'


            day = row['day_feature'] 
            hour = row['hour_feature']
            if day == "" or hour == "":
                prompt+= "\n"+"Check-in Time Information: The POI has no user check-in record. "
            else:
                prompt+= "\n"+"Time Information: People usually visit the POI " + time_dict[hour]+". "
                prompt+= " And people usually come to the POI on " + day +'s.'

            prompt += "\n"+"Question: Where is the POI in " + dataset_city_dict[dataset_name] +"?"
            prompt_result.append(prompt)

            
            
            
            
    elif prompt_type == 'cat_nearby':
        for _, row in tqdm(poi_df.iterrows(), total=poi_df.shape[0]):
            prompt = ""

            name = ""
            name_info = eval(row['osm_names'])
            if "name:en" in name_info:
                name = name_info['name:en']
            elif "name" in name_info:
                name = name_info['name']
            if name !='':
                prompt +=  "The name of the POI is " + name + ". "

            temp = eval(row['coordinates'])
            lon, lat = temp[0], temp[1]
            if lat < 0:
                lat = str(-lat) +" South"
            else:
                lat = str(lat) +" North"
            if lon < 0:
                lon = str(-lon) +" West"
            else:
                lon = str(lon) +" East"

            prompt += prompt_base + lat +" and " + lon + '.'


            category = row['category']
            category_nearby = row['category_nearby']

            prompt += "\n"+"Category Information: The POI is a " + category + "."

            if category_nearby != " ":   
                temp = category_nearby.split(',')
                if len(temp) == 1:
                    prompt += " And there are " + temp[0] + " near the POI."
                elif len(temp) == 2:
                    prompt += " And there are " + temp[0] + ' and ' + temp[1] + " near the POI."
                elif len(temp) == 3:
                    prompt += " And there are " + temp[0] + ', ' + temp[1] + ' and ' + temp[2] + " near the POI."
                

            prompt += "\n"+"Question: Where is the POI in " + dataset_city_dict[dataset_name] +"?"
            prompt_result.append(prompt)

        
    
    save_data_path = "./Prompt/" + "" + dataset_name +"/"+ "prompt_" + dataset_name + "_" + prompt_type + '.csv'
    save_prompt(prompt_result, save_data_path)
            
            
        
if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import torch

from model_init import *
# df = pd.read_csv("./Feature/SG/poi_SG_address.csv")
# df = df.fillna('')
# df['osm_calculated_postcode'] = df['osm_calculated_postcode'].apply(lambda x : str(int(x)) if x!=''  else(''))

# df.to_csv("./Feature/SG/poi_SG_address.csv",index=False)
# path = "Model_state_dict_cache/SG/SG_llama3_ctle_256_Epoch_5_statedict.pt"
# path1 = "Embed/LLM_Embed/SG/SG_llama3_address_LAST.pt"
# path2= "Embed/LLM_Embed/SG/SG_llama3_cat_nearby_LAST.pt"
# path3= "Embed/LLM_Embed/SG/SG_llama2_time_LAST.pt"
# path4= "Embed/Poi_Model_Embed/ctle_256_sg/poi_repr/poi_repr.pth"
# Model = PoiEnhancer(path1, path2, path3, path4)
# Model.load_state_dict(torch.load(path))
# Model.eval()
# import torch
# dataset = 'NY'
# path = "./Dataset/" + dataset + '/'+dataset.lower()+'_geo.csv'
# def  tocafed(x):
#     if x == 'Caf�':
#         return 'Café'
#     elif x =='Caf��':
#         return 'Café'
#     elif x =='Cafés':
#         return 'Café'
#     else: 
#         return x

# poi_df = pd.read_csv(path, sep=',', header=0)
# poi_df['poi_type'] = poi_df['poi_type'].apply(tocafed)
# # poi_df['venue_category_name'] = poi_df['venue_category_name'].apply(tocafed)

# poi_df.drop("Unnamed: 0", axis=1, inplace=True)

# poi_df.to_csv(path)
import pandas as pd

df = pd.read_csv('./Washed_Feature/TKY/poi_TKY_address.csv')
x = 0
df=df.fillna("")
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    housenumber = row['housenumber']
    street = row['street']
    # a = row['full']
    temp_address = ''
    province = row['province']
    city = row['city']
    quarter = row['quarter']
    neighbourhood = row['neighbourhood']
    
    street = row['street']
    blocknumber = row['block_number']
    housenumber = row['housenumber']
    temp_address = province + city + quarter + neighbourhood + blocknumber + housenumber +street
    if  temp_address !='':
        x += 1
print(x)
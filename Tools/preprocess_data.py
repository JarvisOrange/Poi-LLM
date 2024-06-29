import numpy as np
import pandas as pd
from tqdm import tqdm


dataset_path_dict = {
    'NY':'./Dataset/Foursquare_NY/nyc.poitraj',
    'SG':'./Dataset/Foursquare_SG/singapore.poitraj',
    'TKY':'./Dataset/Foursquare_TKY/tky.poitraj',
}

def clean_poi_traj(dataset_name,  save_path=None):
    data_path = dataset_path_dict[dataset_name]
    poi_df = pd.read_csv(data_path, sep=',', header=None)

    columns_standard = ["entity_id","location","time","type","dyna_id"]
    if dataset_name != 'TKY':
        poi_df.columns = columns_standard
    else:
        poi_df.columns = ["dyna_id","type","time","entity_id","location"]
        poi_df = poi_df.loc[:, columns_standard]
    

    poi_cleaned_df = pd.DataFrame(data=None, columns = columns_standard)

    entity_group = poi_df.groupby("entity_id")
    i = 0
    for e_g in tqdm(entity_group):
        single_traj =pd.DataFrame(e_g[1])

        if len(single_traj) < 5:
            print(i)
            i+=1
            continue
        else:
            poi_cleaned_df = pd.concat([poi_cleaned_df, single_traj], ignore_index=True)

       



clean_poi_traj('SG')
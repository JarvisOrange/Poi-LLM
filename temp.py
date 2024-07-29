import numpy as np
import pandas as pd
import torch

from model_init import *
# df = pd.read_csv("./Feature/SG/poi_SG_address.csv")
# df = df.fillna('')
# df['osm_calculated_postcode'] = df['osm_calculated_postcode'].apply(lambda x : str(int(x)) if x!=''  else(''))

# df.to_csv("./Feature/SG/poi_SG_address.csv",index=False)
path = "Model_state_dict_cache/SG/SG_llama3_ctle_256_Epoch_5_statedict.pt"
path1 = "Embed/LLM_Embed/SG/SG_llama3_address_LAST.pt"
path2= "Embed/LLM_Embed/SG/SG_llama3_cat_nearby_LAST.pt"
path3= "Embed/LLM_Embed/SG/SG_llama2_time_LAST.pt"
path4= "Embed/Poi_Model_Embed/ctle_256_sg/poi_repr/poi_repr.pth"
# Model = PoiEnhancer(path1, path2, path3, path4)
# Model.load_state_dict(torch.load(path))
# Model.eval()


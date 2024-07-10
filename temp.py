import numpy as np
import pandas as pd
import torch
# df = pd.read_csv("./Feature/SG/poi_SG_address.csv")
# df = df.fillna('')
# df['osm_calculated_postcode'] = df['osm_calculated_postcode'].apply(lambda x : str(int(x)) if x!=''  else(''))

# df.to_csv("./Feature/SG/poi_SG_address.csv",index=False)
path = "./Embed/Result_Embed/temp.pt"
embed = torch.load(path).to('cuda:2')

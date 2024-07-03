import numpy as np
import pandas as pd

df = pd.read_csv("./Feature/SG/poi_SG_address.csv")
df = df.fillna('')
df['osm_calculated_postcode'] = df['osm_calculated_postcode'].apply(lambda x : str(int(x)) if x!=''  else(''))

df.to_csv("./Feature/SG/poi_SG_address.csv",index=False)
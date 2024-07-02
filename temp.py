import numpy as np
import pandas as pd

df = pd.read_csv("./Feature/TKY/poi_TKY_address.csv")
df.insert(0, 'geo_id', range(len(df)), allow_duplicates=False)

df.to_csv("./Feature/TKY/poi_TKY_address.csv",index=False)

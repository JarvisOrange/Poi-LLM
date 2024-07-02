import numpy as np
import pandas as pd
from tqdm import tqdm


dataset_path_dict = {
    'NY':'./Dataset/Foursquare_NY/ny.poitraj',
    'SG':'./Dataset/Foursquare_SG/sg.poitraj',
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



# import numpy as np
# import pandas as pd
# from sklearn.utils import shuffle

# oldpoidf = pd.read_csv("./data/train_content.txt",sep='\t',header=None)
# oldpoi = list(oldpoidf.iloc[:,0])
# oldpoidict = {}
# for i in range(len(oldpoi)):
#     oldpoidict[oldpoi[i]] = i

# train_df = pd.read_csv("./data/test.txt", sep='\t', header=None)
# train_df.columns = ['usr_id', 'lat', 'long', 'time', 'poi_id', 'text']

# # generate train_context trajectory
# group_user = train_df.groupby('usr_id')
# train_context = []
# train_traj=[]
# train_traj_target = []
# train_data = []
# for traj_group in list(group_user):
#     traj_df = pd.DataFrame(traj_group[1])
#     traj_list = list(traj_df['poi_id'])
#     flag = 1
#     temp_traj_list= []
#     for t in traj_list:
#         if t not in oldpoidict:
#             flag = 0
#         else:
#             temp_traj_list.append(oldpoidict[t])
#     if len(temp_traj_list) >= 9 and flag == 1:
#         train_context.append(traj_list)
#         train_traj.append(temp_traj_list[:9])
#         train_traj_target.append(temp_traj_list[8])

# train_data =[[train_traj[i]] for i in range(len(train_traj_target)) ]

# train_data = pd.DataFrame(train_data)

# train = shuffle(train_data)
# size = int(0.8 * train.shape[0])
# train_data = train.iloc[:size, :]
# test_data = train.iloc[size:, :]

# train_data.to_csv("./train_data/train_traj.csv")
# test_data.to_csv("./test_data/test_traj.csv")






import numpy as np
import pandas as pd

from tqdm import tqdm

import itertools
import random
import random
import os


dataset_path_dict = {
    'NY':'./ContrastDataset/NY/',
    'SG':'./ContrastDataset/SG/',
    'TKY':'./ContrastDataset/TKY/',
}



def gen_contrast_train_data(positive, all, negative_num = 6):
    temp = set(all) - set(positive)
    return  random.sample(temp, negative_num)


def process_contrast_data(dataset_name='NY'):
    data_path = dataset_path_dict[dataset_name]
    geo_positive = pd.read_csv(data_path + dataset_name + "_geo_positive.csv", sep=',', header=0)
    seq_positive = pd.read_csv(data_path + dataset_name + "_seq_positive.csv", sep=',', header=0)
    time_cat_positive = pd.read_csv(data_path + dataset_name + "_time_cat_positive.csv", sep=',', header=0)

    positive_df = pd.merge(geo_positive,time_cat_positive,on='geo_id', how='outer')
    positive_df = pd.merge(positive_df,seq_positive,on='geo_id', how='outer')

    positive_df=positive_df.fillna('[]')

    all_poi = list(positive_df['geo_id'])

    positive_df['geo_positive'] = positive_df['geo_positive'].apply(lambda x: eval(x))

    positive_df['seq_positive'] = positive_df['seq_positive'].apply(lambda x: eval(x))

    positive_df['time_cat_positive'] = positive_df['time_cat_positive'].apply(lambda x: eval(x))
    
    positive_df['positive'] = positive_df['geo_positive'] + positive_df['seq_positive'] + positive_df['time_cat_positive']

    positive_df['positive'] = positive_df['positive'].apply(lambda x: list(set(x)))

    positive_df = positive_df.drop(['geo_positive','seq_positive','time_cat_positive'], axis = 1)


    # genernate negative

    train_data = []
    for _, row in tqdm( positive_df.iterrows(), total= positive_df.shape[0]):
        anchor = row['geo_id']
        positive = row['positive']
        
        for p in list(positive):
            negative = gen_contrast_train_data(positive, all_poi)
            train_data.append([anchor, p, negative] )
        
    train_data_name  = dataset_name + '_train.csv'
    train_data_df  = pd.DataFrame(train_data, columns = ['anchor','positve','negative'])
    train_data_df.to_csv(train_data_name, index=False, header=True, sep=',')
    

process_contrast_data('SG')
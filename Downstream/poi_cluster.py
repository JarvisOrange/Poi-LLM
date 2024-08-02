import torch
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import argparse

device='cuda:0'
def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help="gpu")

    parser.add_argument(
        "--NAME",
        type=str
    )

    parser.add_argument(
        "--POI_MODEL_NAME",
        type=str
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="NY",
        choices=["NY","SG","TKY"],
        help="which dataset",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Result save path",
    )


    args = parser.parse_args()

    return args

if __name__ == '__main__':    
    args = create_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    name = args.NAME
    dataset = args.dataset
    poi_model_name = args.POI_MODEL_NAME

    temp = name.split('_')
    name_without_epoch = '_'.join(temp[:-2])
    embedding = torch.load('Washed_Embed/Result_Embed/{}/{}/{}.pt'.format(dataset,name_without_epoch, name)).to(device)
    category = pd.read_csv('Washed/{}/category.csv'.format(poi_model_name), usecols=['geo_id', 'category'])

    inputs=category.geo_id.to_numpy()
    labels=category.category.to_numpy()
    num_class = labels.max()+1

    node_embedding=embedding[torch.tensor(inputs)].cpu()

    print(f'Start Kmeans, data.shape = {node_embedding.shape}, kinds = {num_class}')
    k_means = KMeans(n_clusters=num_class, random_state=42)
    k_means.fit(node_embedding)
    y_predict = k_means.predict(node_embedding)
    y_predict_useful = y_predict
    nmi = metrics.normalized_mutual_info_score(labels, y_predict_useful)
    ars = metrics.adjusted_rand_score(labels, y_predict_useful)
    # SC指数
    sc = float(metrics.silhouette_score(node_embedding, k_means.labels_, metric='euclidean'))
    # DB指数
    db = float(metrics.davies_bouldin_score(node_embedding, k_means.labels_))
    # CH指数
    ch = float(metrics.calinski_harabasz_score(node_embedding, k_means.labels_))
    print(f"Evaluate result [loc_cluaster] is sc = {sc:6f}, db = {db:6f}, ch = {ch:6f}, nmi = {nmi:6f}, ars = {ars:6f}")
    result = pd.DataFrame({
        'name': args.NAME,
        'sc': sc,
        'db': db,
        'ch': ch,
        'nmi': nmi,
        'ars':ars,
    }, index=[1])

    import os
    save_path = './Washed_Result_Metric/' + args.dataset + '/' + name +'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    result.to_csv(save_path + name + '.cluster', index=False)
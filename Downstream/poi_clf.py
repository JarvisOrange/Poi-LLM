import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from libcity_utils import next_batch
from tqdm import tqdm

import argparse

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

embed_size = 256 # The size of poi embeddings. 128 or 256 in our exp.
task_epoch = 100
downstream_batch_size = 32

if __name__ == '__main__':

    args = create_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    name = args.NAME

    dataset = args.dataset

    path1 = './Embed/Poi_Model_Embed/'+ args.POI_MODEL_NAME+'/poi_repr/'
    path2 = './Embed/Result_Embed/' + dataset + '/'
    category = pd.read_csv(path1 + 'category.csv', usecols=['geo_id', 'category'])
    inputs = torch.load(path2 + name + '.pt').to(device)
    # inputs = torch.load(path1 + "poi_repr.pth").to(device)
    num_loc = len(category)
    labels=category.category.to_numpy()
    indices = list(range(num_loc))
    
    # 随机划分数据集
    np.random.shuffle(indices)
    inputs=inputs[indices]
    labels=labels[indices]
    # 记录num category
    num_class = labels.max()+1
    # 写mlp
    hidden_size = 1024
    clf_model = nn.Sequential(
        nn.Linear(embed_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_class)
    ).to(device)
    
    # optimizer & loss
    optimizer = torch.optim.Adam(clf_model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()
    # kflod test
    skf=StratifiedKFold(n_splits=5)
    score_log = []

    for i,(train_ind,valid_ind) in enumerate(skf.split(inputs,labels)):
        for epoch in tqdm(range(task_epoch), desc=f'Fold {i + 1}/5', total=task_epoch):
            for _, batch in enumerate(next_batch(train_ind, downstream_batch_size)):
                batch_input = inputs[batch].clone()
                batch_label = torch.tensor(labels[batch],dtype=torch.long,device=device)

                out=clf_model(batch_input)
                loss=loss_func(out,batch_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        pres_raw=[]
        test_labels=[]
        for _, batch in enumerate(next_batch(valid_ind, downstream_batch_size)):
            batch_input = inputs[batch].clone()
            batch_label = torch.tensor(labels[batch],dtype=torch.long,device=device)
            out=clf_model(batch_input)

            pres_raw.append(out.detach().cpu().numpy())
            test_labels.append(batch_label.detach().cpu().numpy())

        pres_raw, test_labels = np.concatenate(pres_raw), np.concatenate(test_labels)
        pres = pres_raw.argmax(-1)

        pre = metrics.precision_score(test_labels, pres, average='macro', zero_division=0.0)
        acc, recall = metrics.accuracy_score(test_labels, pres), metrics.recall_score(test_labels, pres, average='macro', zero_division=0.0)
        f1_micro, f1_macro = metrics.f1_score(test_labels, pres, average='micro'), metrics.f1_score(test_labels, pres, average='macro')
        score_log.append([acc, pre, recall, f1_micro, f1_macro])
        print('Acc %.6f, Pre %.6f, Recall %.6f, F1-micro %.6f, F1-macro %.6f' % (
                    acc, pre, recall, f1_micro, f1_macro))
    
    mean_acc, mean_pre, mean_recall, mean_f1_micro, mean_f1_macro = np.mean(score_log, axis=0)
    print('Acc %.6f, Pre %.6f, Recall %.6f, F1-micro %.6f, F1-macro %.6f' % (
        mean_acc, mean_pre, mean_recall, mean_f1_micro, mean_f1_macro))

    result = pd.DataFrame({
            'name': args.NAME,
            'accuracy': mean_acc,
            'precision': mean_pre,
            'recall': mean_recall,
            'f1-micro': mean_f1_micro,
            'f1-macro': mean_f1_macro,
        }, index=[1])
    save_path = './Result_Metric/' + dataset + '/' + name + '.clf'
    result.to_csv(save_path, index=False)
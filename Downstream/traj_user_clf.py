import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from libcity_utils import next_batch, weight_init
import numpy as np
from numpy.random import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

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
        "--embed_size",
        type=int,
        default=128,
        choices=[128, 256],
        help="The embedding size",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help='The ratio of test set in the whole dataset'
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Result save path",
    )

    args = parser.parse_args()

    return args


class LstmUserPredictor(nn.Module):
    def __init__(self, embed_layer, input_size, rnn_hidden_size, fc_hidden_size, output_size, num_layers,device):
        super().__init__()
        self.embed_layer = embed_layer
        self.hidden_size=rnn_hidden_size
        self.num_layers=num_layers
        self.device=device

        self.encoder = nn.LSTM(input_size, rnn_hidden_size, num_layers, dropout=0.1 if num_layers>1 else 0.0, batch_first=True)

        self.fc = nn.Sequential(nn.Tanh(), nn.Linear(rnn_hidden_size, fc_hidden_size),
                                        nn.LeakyReLU(), nn.Linear(fc_hidden_size, output_size))
    
        self.apply(weight_init)

    def forward(self, seq, valid_len, **kwargs):
        
        full_embed = self.embed_layer[seq]
        pack_x = pack_padded_sequence(full_embed, lengths=valid_len,batch_first=True,enforce_sorted=False)

        h0 = torch.zeros(self.num_layers, full_embed.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, full_embed.size(0), self.hidden_size).to(self.device)

        out, _ = self.encoder(pack_x, (h0, c0))
        out, out_len = pad_packed_sequence(out, batch_first=True)

        out = torch.stack([out[i,ind-1,:] for i,ind in enumerate(valid_len)])
        
        pred = self.fc(out)
        return pred


def traj_user_classification(train_set, test_set, num_user, num_loc, clf_model, num_epoch, batch_size, device):
    clf_model = clf_model.to(device)
    optimizer = torch.optim.Adam(clf_model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()

    def one_step(batch):
        user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
        full_seq = [torch.tensor(seq,dtype=torch.long,device=device) for seq in full_seq]
        inputs = pad_sequence(full_seq,batch_first=True, padding_value=num_loc)
        targets = torch.tensor(user_index).long().to(device)
        length = list(length)

        out = clf_model(inputs,length)
        return out, targets

    score_log = []
    test_point = max(1, int(len(train_set) / batch_size / 2))

    for epoch in range(num_epoch):
        for i, batch in enumerate(next_batch(train_set, batch_size)):
            out, label = one_step(batch)
            loss = loss_func(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % test_point == 0:
                pres_raw, labels = [], []
                for test_batch in next_batch(test_set, batch_size * 4):
                    test_out, test_label = one_step(test_batch)
                    pres_raw.append(test_out.detach().cpu().numpy())
                    labels.append(test_label.detach().cpu().numpy())
                pres_raw, labels = np.concatenate(pres_raw), np.concatenate(labels)
                pres = pres_raw.argmax(-1)

                pre = precision_score(labels, pres, average='macro', zero_division=0.0)
                acc, recall = accuracy_score(labels, pres), recall_score(labels, pres, average='macro', zero_division=0.0)
                f1_micro, f1_macro = f1_score(labels, pres, average='micro'), f1_score(labels, pres, average='macro')
                score_log.append([acc, pre, recall, f1_micro, f1_macro])
                best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro = np.max(score_log, axis=0)
                
        print('epoch {} complete!'.format(epoch))
        print('Acc %.6f, Pre %.6f, Recall %.6f, F1-micro %.6f, F1-macro %.6f' % (
                    best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro))

    best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro = np.max(score_log, axis=0)
    print('Finished Evaluation.')
    print(
        'Acc %.4f, Pre %.4f, Recall %.4f, F1-micro %.4f, F1-macro %.4f' % (
            best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro))
    return best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro

if __name__ == '__main__':
    args = create_args()
    hidden_size = 512
    device = torch.device(f"cuda:{args.gpu}" if args.gpu and torch.cuda.is_available() else "cpu")

    category = pd.read_csv(f'PATH_TO_DATASET/category.csv', usecols=['geo_id'])
    num_loc = len(category)

    poi_embedding = torch.load(f'PATH_TO_DATASET/poi_repr.pth').to(device)

    traj_set =  torch.load(f'PATH_TO_DATASET/traj_set.pth')
    whole_set = list(filter(lambda data: len(data[1]) > 5, traj_set))

    unique_uids = np.unique([data[0] for data in whole_set])
    num_user = len(unique_uids)
    for data in whole_set:
        assert data[0] in unique_uids
        data[0] = np.searchsorted(unique_uids, data[0])

    np.random.seed(42)
    shuffle(whole_set)
    train_set = whole_set[int(len(whole_set) * args.test_ratio):]
    test_set = whole_set[:int(len(whole_set) * args.test_ratio)]
    print(f'Train set size: {len(train_set)}, test set size: {len(test_set)}')
    
    unique_uids = np.unique([data[0] for data in whole_set])
    num_user = len(unique_uids)
    assert unique_uids.max() + 1 == num_user
    

    downstream_batch_size = 32
    epoch = 100

    clf_model=LstmUserPredictor(poi_embedding, args.embed_size,hidden_size,hidden_size,num_user,num_layers=2,device=device)

    result = {'name': args.NAME}
    result['traj_clf_acc'], result['traj_clf_pre'], result['traj_clf_recall'], result['traj_clf_f1_micro'], result['traj_clf_f1_macro'] =\
    traj_user_classification(train_set, test_set, num_user, num_loc, clf_model,
                    num_epoch=epoch, batch_size=downstream_batch_size, device=device)
    pd.DataFrame(result, index=[1]).to_csv(args.save_path, index=False)

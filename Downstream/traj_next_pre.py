import pandas as pd
import numpy as np
import torch
from torch import nn
from libcity_utils import *
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn import metrics
from numpy.random import shuffle
from tqdm import tqdm

embed_size = 256 # The size of poi embeddings. 128 or 256 in our exp.
task_epoch = 50
downstream_batch_size = 32
pre_model_seq2seq = True
predict_len = 1
test_ratio = 0.4

import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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

def seq2seq_forward(encoder, lstm_input, valid_len, pre_len):
    his_len = valid_len - pre_len
    src_padded_embed = pack_padded_sequence(lstm_input, his_len.cpu(), batch_first=True, enforce_sorted=False)
    out, hc = encoder(src_padded_embed)
    out,out_len=pad_packed_sequence(out,batch_first=True)
    return out,out_len
            


class TrajectoryPredictor(nn.Module):
    def __init__(self, num_slots, aux_embed_size, time_thres, dist_thres,
                 input_size, lstm_hidden_size, fc_hidden_size, output_size, num_layers, seq2seq=True):
        super().__init__()
        self.__dict__.update(locals())

        self.time_embed = nn.Embedding(num_slots + 1, aux_embed_size)
        self.dist_embed = nn.Embedding(num_slots + 1, aux_embed_size)

        self.encoder = nn.LSTM(input_size + 2 * aux_embed_size, lstm_hidden_size, num_layers, dropout=0.3,
                               batch_first=True)
        self.ln = nn.LayerNorm(lstm_hidden_size)
        self.out_linear = nn.Sequential(nn.Tanh(), nn.Linear(lstm_hidden_size, fc_hidden_size),
                                        nn.Tanh(), nn.Linear(fc_hidden_size, output_size))
        self.sos = nn.Parameter(torch.zeros(input_size + 2 * aux_embed_size).float(), requires_grad=True)
        self.aux_sos = nn.Parameter(torch.zeros(aux_embed_size * 2).float(), requires_grad=True)
        self.apply(weight_init)

        

    def forward(self, full_embed, valid_len, pre_len, **kwargs):
        batch_size = full_embed.size(0)
        # his_len = valid_len - pre_len

        time_delta = kwargs['time_delta'][:, 1:]
        dist = kwargs['dist'][:, 1:]

        time_slot_i = torch.floor(torch.clamp(time_delta, 0, self.time_thres) / self.time_thres * self.num_slots).long()
        dist_slot_i = torch.floor(
            torch.clamp(dist, 0, self.dist_thres) / self.dist_thres * self.num_slots).long()  # (batch, seq_len-1)
        aux_input = torch.cat([self.aux_sos.reshape(1, 1, -1).repeat(batch_size, 1, 1),
                               torch.cat([self.time_embed(time_slot_i),
                                          self.dist_embed(dist_slot_i)], dim=-1)],
                              dim=1)  # (batch, seq_len, aux_embed_size*2)
        lstm_input = torch.cat([full_embed, aux_input],
                               dim=-1)  # (batch_size, seq_len, input_size + aux_embed_size * 2)

        if self.seq2seq:
            lstm_out_pre,out_len = seq2seq_forward(self.encoder, lstm_input, valid_len, pre_len)
        else:
            raise NotImplementedError()
            lstm_out_pre = rnn_forward(self.encoder, self.sos, lstm_input, valid_len, pre_len)

        lstm_out_pre=self.ln(lstm_out_pre)
        out = self.out_linear(lstm_out_pre)
        return out

def one_step(pre_model, pre_len, embedding, num_loc, batch):
    def _create_src_trg(origin, fill_value):
        src, trg = create_src_trg(origin, pre_len, fill_value)
        return torch.from_numpy(src).float().to(device)


    user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
    index_matrix=torch.zeros([len(length),max(length)-1],dtype=torch.bool)
    for i in range(len(length)):
        index_matrix[i][:length[i]-1]=~index_matrix[i][:length[i]-1]
    index_matrix=index_matrix.to(device)
    user_index, length = (torch.tensor(item).long().to(device) for item in (user_index, length))

    src_seq, trg_seq = create_src_trg(full_seq, pre_len, fill_value=num_loc)

    src_seq, trg_seq = (torch.from_numpy(item).long().to(device) for item in [src_seq, trg_seq])

    src_t = _create_src_trg(timestamp, 0)
    src_time_delta = _create_src_trg(time_delta, 0)
    src_dist = _create_src_trg(dist, 0)
    src_lat = _create_src_trg(lat, 0)
    src_lng = _create_src_trg(lng, 0)

    src_seq_embedded = []
    for seq in src_seq.cpu():
        seq_embedded = []
        for elem in seq:
            seq_embedded.append(torch.zeros(embed_size, device=device) if elem.item() == num_loc else embedding[elem])
        src_seq_embedded.append(torch.stack(seq_embedded))
    src_seq_embedded = torch.stack(src_seq_embedded)
    out = pre_model(src_seq_embedded, length, pre_len, user_index=user_index, timestamp=src_t,
                    time_delta=src_time_delta, dist=src_dist, lat=src_lat, lng=src_lng)

    out = out[index_matrix]
    label = trg_seq[index_matrix]
    return out, label

if __name__ == '__main__':

    args = create_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    name = args.NAME

    dataset = args.dataset

    path1 = './Washed/'+ args.POI_MODEL_NAME+'/'

    temp = name.split('_')

    name_without_epoch = '_'.join(temp[:-2])
    
    path2 = './Washed_Embed/Result_Embed/' + dataset + '/' + name_without_epoch +'/'


    
    #FIXME
    category = pd.read_csv(path1+'category.csv', usecols=['geo_id'])
    
    traj_set = torch.load(path1+'traj_set.pth')


    poi_embedding = torch.load(path2 + name +'.pt').to(device)

        
    

    #We have to remake the train set and test set
    whole_set = list(filter(lambda data: len(data[1]) > predict_len, traj_set))
    np.random.seed(42)
    shuffle(whole_set)
    train_set = whole_set[int(len(whole_set) * test_ratio):]
    test_set = whole_set[:int(len(whole_set) * test_ratio)]
    print(f'Train set size: {len(train_set)}, test set size: {len(test_set)}')
    

    num_loc = len(category)
    pre_model = TrajectoryPredictor(num_slots=10, aux_embed_size=16,
                                    time_thres=10800, dist_thres=0.1,
                                    input_size=embed_size, lstm_hidden_size=512,
                                    fc_hidden_size=embed_size * 4, output_size=num_loc, num_layers=2,
                                    seq2seq=pre_model_seq2seq)


    print('Start training downstream model...')
    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()


    score_log = []
    test_point = max(1, int(len(train_set) / downstream_batch_size / 2))
    for epoch in range(task_epoch):
        losses=[]
        for i, batch in enumerate(next_batch(train_set, downstream_batch_size)):
            out, label = one_step(pre_model, predict_len, poi_embedding, num_loc, batch)
            loss = loss_func(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (i + 1) % test_point == 0:
                with torch.no_grad():
                    pres_raw, labels = [], []
                    for test_batch in next_batch(test_set, downstream_batch_size * 4):
                        test_out, test_label = one_step(pre_model, predict_len, poi_embedding, num_loc, test_batch)
                        pres_raw.append(test_out.detach().cpu())
                        labels.append(test_label.detach().cpu())
                    pres_raw, labels = torch.vstack(pres_raw), torch.hstack(labels)
                    pres = pres_raw.argmax(-1)

                    acc1,acc5 = accuracy(pres_raw,labels,topk=(1,5)) 
                    f1_micro, f1_macro = metrics.f1_score(labels.numpy(), pres.numpy(), average='micro'), metrics.f1_score(labels.numpy(), pres.numpy(), average='macro')
                    score_log.append([acc1, acc5, f1_micro, f1_macro])
                    print('Acc@1 %.6f, Acc@5 %.6f, F1-micro %.6f, F1-macro %.6f' % (
                    acc1, acc5, f1_micro, f1_macro))
                    best_acc1, best_acc5, best_f1_micro, best_f1_macro = np.max(score_log, axis=0)
        
        if epoch % 5 == 0:
            result = pd.DataFrame({
                'name': args.NAME,
                'accuracy1': best_acc1,
                'accuracy5': best_acc5,
                'f1-micro': best_f1_micro,
                'f1-macro': best_f1_macro,
                'epoch': epoch
            }, index=[1])

            
            save_path = './Washed_Result_Metric/' + args.dataset + '/' + name +'/'
            if not os.path.exists(save_path):
                    os.makedirs(save_path)

            result.to_csv(save_path + args.NAME + '.pre', index=False)
                    
        print('epoch {} complete! avg loss:{}'.format(epoch,np.mean(losses)))

    
    print('Finished Evaluation.')
    print(
        'Acc1 %.6f %%, Acc5 %.6f %%, F1-micro %.6f, F1-macro %.6f' % (
            best_acc1, best_acc5, best_f1_micro, best_f1_macro))

    


    result = pd.DataFrame({
        'name': args.NAME,
        'accuracy1': best_acc1,
        'accuracy5': best_acc5,
        'f1-micro': best_f1_micro,
        'f1-macro': best_f1_macro,
    }, index=[1])

    import os
    save_path = './Washed_Result_Metric/' + args.dataset + '/' + name +'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)

    result.to_csv(save_path + name + '.pre', index=False)
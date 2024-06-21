import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils import data
from torch.utils.data import DataLoader	
from torch.nn import init
from tqdm import *
import numpy as np
from utils import weight_init
import torch.nn.init as init
import os
from info_nce import InfoNCE, info_nce
from torch.nn import TripletMarginLoss
from utils import SimliarityLoss
from utils import TeacherStudentLoss
import argparse
import logging




class TrainDataset(data.Dataset):
    def __init__(self,path):
        self.data = pd.read_csv(path, sep=',', header=1)
        
        self.traj = [eval(x) for x in list(self.data.iloc[:,1])]
        
        
    def __getitem__(self, index):
        traj = torch.Tensor(self.traj[index][:8]).cuda()
        target = self.traj[index][8]
        label = torch.zeros(poi_size).cuda()
        label[target] = 1
        return traj, label
    
    def __len__(self):
        return self.data.shape[0]
    
class TestTrajDataset(data.Dataset):
    def __init__(self,path):
        self.data = pd.read_csv(path, sep=',', header=1)
        
        self.traj = [eval(x) for x in list(self.data.iloc[:,1])]
        
    def __getitem__(self, index):
        traj = torch.Tensor(self.traj[index][:8]).cuda()
        target = self.traj[index][8]
        label = torch.zeros(poi_size).cuda()
        label[target] = 1
        return traj, label,target
    
    def __len__(self):
        return self.data.shape[0]
BATCH_SIZE = 128
EPOCH = 10
lr = 0.01

if __name__ == '__main__':
    pass
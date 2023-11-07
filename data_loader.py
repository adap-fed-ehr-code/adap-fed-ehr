# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset


class WholeDataLoader(Dataset):
    def __init__(self,X,y,d):
        self.X,self.y,self.d = torch.from_numpy(X),torch.from_numpy(y),torch.from_numpy(d)

    def __getitem__(self,index):
        return self.X[index,:],self.y[index,None],self.d[index,None]
        # return torch.from_numpy(x),torch.from_numpy(np.array([y])),torch.from_numpy(np.array([z]))

    def __len__(self):
        return self.X.shape[0]




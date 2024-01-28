import numpy as np
import warnings
import h5py
import torch.utils.data as data
warnings.filterwarnings('ignore')
import torch


datapath = 'data/MnistScatData.h5'

class ScaDataset(data.Dataset):
    def __init__(self, datapath, split='train'):
        if  split == 'train':
            self.data = h5py.File(datapath, 'r')['train']

        else:
            self.data = h5py.File(datapath, 'r')['test']

    def __getitem__(self, index):
        esct = self.data['Esct'][index].reshape(216, 1)  #216
        pce = self.data['points'][index]  #2048 4
 
        return esct, pce     

    def __len__(self):
        return len(self.data['Esct'])

if __name__ == '__main__':
    train = ScaDataset(datapath=datapath, split='train')

    for sct, points in train:
        print(sct.shape, type(sct))
        print(points.shape, type(points))
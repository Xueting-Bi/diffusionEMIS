import numpy as np
import warnings
import h5py
import torch.utils.data as data
warnings.filterwarnings('ignore')
import torch



point_path = '/home/xueting/data/prl/second/mwpnor5000.hdf5'
wave_path = '/home/xueting/data/prl/second/1_micro.npy'
class MwpDataset(data.Dataset):
    def __init__(self, proot=point_path, wroot=wave_path, split='train'):
        
        self.proot = proot
        self.wroot = wroot
        data = h5py.File(self.proot)
        mw = np.load(self.wroot) 
        mw = mw.reshape(5000, 1, 2560)

        
        if split == 'train':
            self.pcs = data['19950406']['train']
            self.waves = mw[:4500]
            
        else:
            self.pcs = data['19950406']['test']
            self.waves = mw[4500:]            
        

    def __getitem__(self, index):
        pc = self.pcs[index]
        wave = self.waves[index]
        np.random.shuffle(pc)
        return torch.Tensor(pc), wave
        

    def __len__(self):
        return len(self.pcs)
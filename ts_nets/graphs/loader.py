import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

DATA_PATH = '../data/'

class GraphDataset(Dataset): 
    
    def __init__(self,
                args,
                flag='train'):
        
        self.root_path = DATA_PATH
        self.X_filename = f'seq{args.seq_len}_tar{args.pred_len}_X.npy'
        self.y_filename = f'seq{args.seq_len}_tar{args.pred_len}_y.npy'
        self.offense_only = args.offense_only     
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self._read_data()

    def _read_data(self):
        self.data_x = np.load(os.path.join(self.root_path, self.X_filename))
        self.data_y = np.load(os.path.join(self.root_path, self.y_filename))

        # Columns order :
        ### 0 : action id
        ### 1 : target
        ### 2 : offense
        
        # Offense filter
        if self.offense_only :
            off_mask = self.data_x[:,0,2].astype(bool)
            self.data_x = self.data_x[off_mask]
            self.data_y = self.data_y[off_mask]

        # Only keeping the player features and the target, and reshaping X in 4D
        target_index = 1 #Action progression is in the second column on the last dimension                      
        self.data_x,self.data_y  = self.data_x[:, :,-60:], self.data_y[:, -1, target_index]
        self.data_x = self.data_x.reshape(self.data_x.shape[0], self.data_x.shape[1], 15, -1)
            
        # TDO : makes this parametrizable
        train_share, val_share, test_share = 0.7, 0.15, 0.15 
        n = self.data_x.shape[0]
        split_indices = [0 , int(np.floor(n * train_share)), int(np.floor(n * (train_share + val_share))), n] 
        
        # Train / val / test split
        self.data_x = self.data_x[split_indices[self.set_type] : split_indices[self.set_type +1]]
        self.data_y = self.data_y[split_indices[self.set_type] : split_indices[self.set_type +1]]


    def __getitem__(self, index):

        return torch.tensor(self.data_X[index], dtype=torch.float32), torch.tensor(self.data_y[index], dtype=torch.float32)

    def __len__(self):
        
        return len(self.data_x)

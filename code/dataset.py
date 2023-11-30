import torch
import torch.nn as nn
from torch.utils.data import Dataset


# dataとlabelをtensorのイテレータに変換するクラス
class MyDataset(Dataset):
    def __init__(self, data, label):
        super(MyDataset, self).__init__()

        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).float()
        label = torch.from_numpy(self.label[idx]).float()

        return data, label
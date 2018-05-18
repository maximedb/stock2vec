import random
from torch.utils.data import Dataset


class skipDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        return row['input'], row['target']


class skipDatasetNegativeSampling(Dataset):

    def __init__(self, data, num_negs):
        self.data = data
        self.num_negs = num_negs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        noise = random.sample(row['noise'], self.num_negs)
        return row['input'], row['target'], noise


def collate_fn(data):
    return zip(*data)


class gloveDataset(Dataset):

    def __init__(self, indices, correlations, s2i):
        self.indices = indices
        self.correlations = correlations
        self.s2i = s2i

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        left, right = self.indices[idx]
        corr = self.correlations.loc[left, right]
        return self.s2i[left], self.s2i[right], corr

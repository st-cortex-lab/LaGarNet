import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from agent_arena import TrajectoryDataset

class CountBasedDataset(TrajectoryDataset):

    def __init__(self, alpha=0.7, beta=0.4, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.counts = np.ones(self.num_samples, dtype=np.float32)
        #print('size priorities buffer', self.priorities.shape)

    def update_counts(self, indices):
        #print('priorities', priorities)
        # print('indices', indices)
        self.counts[indices] += 1
    
    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        res['idx'] = idx
        return res
    
class CountBasedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        probs = 1/self.dataset.counts[:self.dataset.num_samples]
        #print('probs len', len(probs))
        probs /= probs.sum()
        indices = np.random.choice(self.dataset.num_samples, self.batch_size, p=probs, replace=False)
        return iter(indices.tolist())

    
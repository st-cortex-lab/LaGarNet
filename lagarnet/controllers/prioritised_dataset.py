import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from actoris_harena import TrajectoryDataset

class PrioritisedTrajectoryDataset(TrajectoryDataset):

    def __init__(self, alpha=0.7, beta=0.4, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.priorities =  np.ones(self.num_samples, dtype=np.float32)*1000
        #print('size priorities buffer', self.priorities.shape)

    def update_priorities(self, indices, priorities):
        priorities = np.asarray(priorities)
        priorities = (priorities + 1e-5) ** self.alpha
        #print('priorities', priorities)
        # print('indices', indices)
        self.priorities[indices] = priorities
    
    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        res['idx'] = idx
        return res
    
    def get_weights(self, indices):
        probs = self.priorities[indices] / self.priorities[:self.num_samples].sum()
        weights = (self.num_samples * probs) ** -self.beta
        return torch.from_numpy(weights / weights.max()).float()


class PrioritisedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        probs = self.dataset.priorities[:self.dataset.num_samples]
        #print('probs len', len(probs))
        probs /= probs.sum()
        indices = np.random.choice(self.dataset.num_samples, self.batch_size, p=probs, replace=False)
        return iter(indices.tolist())

    
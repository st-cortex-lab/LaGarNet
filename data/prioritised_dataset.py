import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from agent_arena import TrajectoryDataset

class PrioritisedTrajectoryDataset(TrajectoryDataset):

    def __init__(self, alpha=0.7, beta=0.4, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.priorities =  np.ones(self.num_samples, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        priorities = np.asarray(priorities)
        self.priorities[indices] = (priorities + 1e-5) ** self.alpha
    
    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        res['idx'] = idx
        return res


class PrioritisedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        probs = self.dataset.priorities[:self.dataset.num_samples]
        probs /= probs.sum()
        indices = np.random.choice(self.dataset.num_samples, self.batch_size, p=probs)
        return iter(indices.tolist())

    def get_weights(self, indices):
        probs = self.dataset.priorities[indices] / self.dataset.priorities[:self.dataset.size].sum()
        weights = (self.dataset.size * probs) ** -self.dataset.beta
        return torch.from_numpy(weights / weights.max()).float()
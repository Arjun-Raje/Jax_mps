from torchvision.transforms import ToTensor, Resize, Lambda, Compose
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import torch

def numpy_collate(batch):
    if isinstance(batch, list):    
        return np.stack([x[0] for x in batch]), np.array([x[1] for x in batch])
    return np.array(batch)

class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)
def transform(x):
    return torch.cat([torch.cos(torch.pi/2 * x), torch.sin(torch.pi/2 * x)], 0).reshape(2,-1).T

def get_data():
    training_data = MNIST(root="data", train=True, download=True, transform=Compose([ToTensor(), 
                                                                                        Lambda(transform)]))
    test_data = MNIST(root="data", train=False, download=True,transform=Compose([ToTensor(), 
                                                                                    Lambda(transform)]))
    train_dataloader = NumpyLoader(training_data, batch_size=1, shuffle=True)
    test_dataloader = NumpyLoader(test_data, batch_size=1, shuffle=True)
    
    return train_dataloader, test_dataloader

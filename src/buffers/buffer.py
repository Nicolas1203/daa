from copyreg import pickle
import torch
import numpy as np
import random as r
import logging as lg

from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms

from src.utils.data import get_color_distortion
from src.utils.utils import timing, get_device
from src.datasets.memory import MemoryDataset

class Buffer(torch.nn.Module):
    def __init__(self, max_size=200, shape=(3,32,32), n_classes=10):
        super().__init__()
        self.n_classes = n_classes  # For print purposes only
        self.max_size = max_size
        self.shape = shape
        self.n_seen_so_far = 0
        self.n_added_so_far = 0
        self.device = get_device()
        if self.shape is not None:
            if len(self.shape) == 3:
                self.register_buffer('buffer_imgs', torch.FloatTensor(self.max_size, self.shape[0], self.shape[1], self.shape[2]).fill_(0))
            elif len(self.shape) == 1:
                self.register_buffer('buffer_imgs', torch.FloatTensor(self.max_size, self.shape[0]).fill_(0))
        self.register_buffer('buffer_labels', torch.LongTensor(self.max_size).fill_(-1))

    def update(self, imgs, labels=None):
        raise NotImplementedError

    def stack_data(self, img, label):
        if self.n_seen_so_far < self.max_size:
            self.buffer_imgs[self.n_seen_so_far] = img
            self.buffer_labels[self.n_seen_so_far] = label
            self.n_added_so_far += 1

    def replace_data(self, idx, img, label):
        self.buffer_imgs[idx] = img
        self.buffer_labels[idx] = label
        self.n_added_so_far += 1
    
    def is_empty(self):
        return self.n_added_so_far == 0
    
    def random_retrieve(self, n_imgs=100):
        if self.n_added_so_far < n_imgs:
            lg.debug(f"""Cannot retrieve the number of requested images from memory {self.n_added_so_far}/{n_imgs}""")
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far]
        
        ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), n_imgs)
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]
        
        return ret_imgs, ret_labels
    
    def bootstrap_retrieve(self, n_imgs=100):
        if self.n_added_so_far == 0:
            return torch.Tensor(), torch.Tensor() 
        ret_indexes = [r.randint(0, min(self.n_added_so_far, self.max_size)-1) for _ in range(n_imgs)]            
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]

        return ret_imgs, ret_labels
        
    def n_data(self):
        return len(self.buffer_labels[self.buffer_labels >= 0])

    def get_all(self):
        return self.buffer_imgs[:min(self.n_added_so_far, self.max_size)],\
             self.buffer_labels[:min(self.n_added_so_far, self.max_size)]

    def get_indexes_of_class(self, label):
        return torch.nonzero(self.buffer_labels == label)
    
    def get_indexes_out_of_class(self, label):
        return torch.nonzero(self.buffer_labels != label)

    def is_full(self):
        return self.n_data() == self.max_size

    def get_labels_distribution(self):
        np_labels = self.buffer_labels.numpy().astype(int)
        counts = np.bincount(np_labels[self.buffer_labels >= 0], minlength=self.n_classes)
        tot_labels = len(self.buffer_labels[self.buffer_labels >= 0])
        if tot_labels > 0:
            return counts / len(self.buffer_labels[self.buffer_labels >= 0])
        else:
            return counts

    def get_major_class(self):
        np_labels = self.buffer_labels.numpy().astype(int)
        counts = np.bincount(np_labels[self.buffer_labels >= 0])
        return counts.argmax()

    def get_max_img_per_class(self):
        n_classes_in_memory = len(self.buffer_labels.unique())
        return int(len(self.buffer_labels[self.buffer_labels >= 0]) / n_classes_in_memory)

import torch
import time
import torch.nn as nn
import random as r
import numpy as np
import os
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

from src.learners.supcon import SupConLearner
from src.utils.losses import SupConLoss


class SimCLRLearner(SupConLearner):
    """Just SimCLR (without any memory)
    """
    def __init__(self, args):
        super().__init__(args)

    def train(self, dataloader, epoch, **kwargs):
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, _ = batch[0].to(self.device), batch[1].to(self.device)
            self.stream_idx += len(batch_x)
            
            # Augment
            batch_aug1 = self.transform_train(batch_x)
            batch_aug2 = self.transform_train(batch_x)

            # Inference
            _, projections1 = self.model(batch_aug1)  # (batch_size, projection_dim)
            _, projections2 = self.model(batch_aug2)
            projections = torch.cat([projections1.unsqueeze(1), projections2.unsqueeze(1)], dim=1)

            # Loss
            loss = self.criterion(features=projections, labels=None)
            self.loss = loss.item()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            # Plot to tensorboard
            if self.params.tensorboard:
                self.plot()

            if ((not j % self.params.test_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                print(
                    f"Epoch : {epoch}   batch {j}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )
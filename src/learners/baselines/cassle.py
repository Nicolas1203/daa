import torch
import time
import torch.nn as nn
import sys
import logging as lg

from torch.utils.data import DataLoader
from copy import deepcopy

from src.learners.base import BaseLearner
from src.utils.losses import CaSSLELoss
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match

class CASSLELearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )
        self.distill_predictor = nn.Sequential(
                nn.Linear(128, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 128),
            ).to(self.device)
        self.learnable_parameters = list(self.model.parameters()) + list(self.distill_predictor.parameters())
        self.optim = torch.optim.SGD(
            self.learnable_parameters,
            lr=self.params.learning_rate,
            momentum=self.params.momentum,
            weight_decay=self.params.weight_decay
            )
        self.cont_loss = SupConLoss(self.params.temperature)

    def init_tag(self):
        """How to save exp name. ex : m200mbs100sbs10
        """
        self.params.tag += f"_m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}"
        lg.info(f"Using the following tag for this experiment : {self.params.tag}")

    def load_criterion(self):
        return CaSSLELoss(self.params.temperature) 

    def train(self, dataloader, task_name, **kwargs):
        self.model = self.model.train()
        self.distill_predictor.train()
        task_id = kwargs.get('task_id', None)
        if task_id > 0:
            self.frozen_model = deepcopy(self.model)

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Combined batch
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)

                    # Augment
                    combined_aug1 = self.transform_train(combined_x)
                    combined_aug2 = self.transform_train(combined_x)

                    self.model.train()

                    # Inference
                    _, z1 = self.model(combined_aug1)
                    _, z2 = self.model(combined_aug2)

                    if task_id > 0:
                        _, z1_frozen = self.frozen_model(combined_aug1.detach())
                        _, z2_frozen = self.frozen_model(combined_aug2.detach())
                    else:
                        _, z1_frozen = self.model(combined_aug1.detach())
                        _, z2_frozen = self.model(combined_aug2.detach())

                    p1 = self.distill_predictor(z1)
                    p2 = self.distill_predictor(z2)

                    # Loss
                    loss_contrastive = self.cont_loss(features=torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1))
                    loss_distill = self.criterion(p1, p2, z1_frozen, z2_frozen)

                    loss = loss_contrastive + loss_distill
                    loss = loss.mean()
                    self.loss = loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    
            # Update buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            if ((not j % self.params.test_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
                self.save(model_name=f"ckpt_{task_name}.pth")
    
    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y

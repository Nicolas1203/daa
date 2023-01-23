import torch
import time
import torch.nn as nn
import sys
import logging as lg

from torch.utils.data import DataLoader

from src.learners.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match

class SCRLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # In SCR they use the images from memory for evaluation
        self.params.eval_mem = True
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )

    def init_tag(self):
        """How to save exp name. ex : m200mbs100sbs10
        """
        self.params.tag += f"_m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}"
        lg.info(f"Using the following tag for this experiment : {self.params.tag}")

    def load_criterion(self):
        return SupConLoss(self.params.temperature) 

    def train(self, dataloader, task_name, **kwargs):
        self.model = self.model.train()

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
                    f1, p1 = self.model(combined_aug1)
                    f2, p2 = self.model(combined_aug2)

                    # features = torch.cat([f1_mem.unsqueeze(1), f2_mem.unsqueeze(1)], dim=1)
                    projections = torch.cat([
                        p1.unsqueeze(1),
                        p2.unsqueeze(1),
                        ], 
                        dim=1
                        )
                    # Loss
                    loss = self.criterion(features=projections, labels=combined_y if self.params.supervised else None)
                    loss = loss.mean()
                    self.loss = loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    
            # Update buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            # Plot to tensorboard
            if self.params.tensorboard:
                self.plot()

            if ((not j % self.params.test_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
                self.save(model_name=f"ckpt_{task_name}.pth")
    
    def plot(self):
        self.writer.add_scalar("loss", self.loss, self.stream_idx)
        self.writer.add_scalar("n_added_so_far", self.buffer.n_added_so_far, self.stream_idx)
        percs = self.buffer.get_labels_distribution()
        for i in range(self.params.n_classes):
            self.writer.add_scalar(f"labels_distribution/{i}", percs[i], self.stream_idx)

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y

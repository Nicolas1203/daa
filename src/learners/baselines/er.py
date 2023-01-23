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

from src.models.resnet import SupConResNet
from src.learners.ce import CELearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils.metrics import forgetting_line   


class ERLearner(CELearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = Reservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )

    def init_tag(self):
        """How to save exp name. ex : m200mbs100sbs10
        """
        self.params.tag += f"_m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}e{self.params.epochs}"
        print(f"Using the following tag for this experiment : {self.params.tag}")

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
                    if self.params.aug: combined_x = self.transform_train(combined_x)

                    # Inference
                    _, logits = self.model(combined_x)

                    # Loss
                    loss = self.criterion(logits, combined_y.long())
                    self.loss = loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
            
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            # Plot to tensorboard
            if self.params.tensorboard:
                self.plot()

            if ((not j % self.params.test_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )

    def evaluate_offline(self, dataloaders, epoch):
        with torch.no_grad():
            self.model.eval()

            test_preds, test_targets = self.encode(dataloaders['test'])
            acc = accuracy_score(test_preds, test_targets)
            self.results.append(acc)
            
            if not ((epoch + 1) % 5):
                self.save(model_name=f"CE_{epoch}.pth")
        
        print(f"ACCURACY {self.results[-1]}")
        return self.results[-1]

    def encode(self, dataloader, nbatches=-1):
        i = 0
        with torch.no_grad():
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(self.device)
                _, logits = self.model(self.transform_test(inputs))
                _, preds = torch.max(logits, 1)
                
                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat = preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat = np.hstack([all_feat, preds.cpu().numpy()])
        return all_feat, all_labels

    def save_results_offline(self):
        if self.params.run_id is not None:
            results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.run_id}")
        else:
            results_dir = os.path.join(self.params.results_root, self.params.tag)

        print(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        pd.DataFrame(self.results).to_csv(os.path.join(results_dir, 'acc.csv'), index=False)

        self.save_parameters()

    def plot(self):
        self.writer.add_scalar("loss", self.loss, self.stream_idx)
    
    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y

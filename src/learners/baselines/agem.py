import torch
import time
import torch.nn as nn
import logging as lg

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from src.learners.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.models.resnet import SupConResNet
from src.learners.ce import CELearner


class AGEMLearner(CELearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )
        self.model = SupConResNet(
            head='mlp',
            dim_in= 640 if self.params.dataset == 'tiny' else 160,
            proj_dim=self.params.n_classes,
            input_channels=self.params.nb_channels
        ).to(self.device)
        self.optim = self.load_optim()

    def init_tag(self):
        """How to save exp name. ex : m200mbs100sbs10
        """
        self.params.tag += f"_m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}"
        lg.info(f"Using the following tag for this experiment : {self.params.tag}")
    
    def load_criterion(self):
        return nn.CrossEntropyLoss()

    def train(self, dataloader, task_name, **kwargs):
        task_id = kwargs.get('task_id', None)
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0].to(self.device), batch[1].long().to(self.device)
            self.stream_idx += len(batch_x)

            for _ in range(self.params.mem_iters):
                _, logits = self.model(self.transform_train(batch_x))

                # Loss
                loss = self.criterion(logits, batch_y)
                self.loss = loss.item()
                # Optim
                self.optim.zero_grad()
                loss.backward()

                if task_id > 0:
                    # sample from memory of previous tasks
                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    if mem_x.size(0) > 0:
                        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                        # gradient computed using current batch
                        grad = [p.grad.clone() for p in params if p.grad is not None]
                        mem_x = mem_x.to(self.device)
                        mem_y = mem_y.to(self.device)
                        _, mem_logits = self.model(self.transform_train(mem_x))
                        loss_mem = self.criterion(mem_logits, mem_y)
                        self.optim.zero_grad()
                        loss_mem.backward()
                        # gradient computed using memory samples
                        grad_ref = [p.grad.clone() for p in params if p.grad is not None]

                        # inner product of grad and grad_ref
                        prod = sum([torch.sum(g * g_r) for g, g_r in zip(grad, grad_ref)])
                        if prod < 0:
                            prod_ref = sum([torch.sum(g_r ** 2) for g_r in grad_ref])
                            # do projection
                            grad = [g - prod / prod_ref * g_r for g, g_r in zip(grad, grad_ref)]
                        # replace params' grad
                        for g, p in zip(grad, params):
                            if p.grad is not None:
                                p.grad.data.copy_(g)
                self.optim.step()
            # Update buffer
            self.buffer.update(imgs=batch_x, labels=batch_y)

            if ((not j % self.params.test_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
                self.save(model_name=f"ckpt_{task_name}.pth")
import torch
import time
import logging as lg

from src.learners.base import BaseLearner
from src.utils.losses import SupConLoss
from src.utils.augment import StyleAugment, MixUpAugment, CutMixAugment
from src.utils import name_match


class AugLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )
        self.tf_seq = {}
        for i in range(self.params.n_augs):
            self.tf_seq[f"aug{i}"] = self.transform_train
        if self.params.n_styles >= 1:
            self.tf_seq["style"] = StyleAugment(
                input_size=self.params.img_size,
                transfer_size=self.params.tf_size,
                parallel=self.params.parallel,
                samples=self.params.style_samples,
                min_alpha=self.params.min_style_alpha,
                max_alpha=self.params.max_style_alpha
                )
        # Legacy condition. To update.
        if self.params.mixup or self.params.n_mixup >= 1:
            self.tf_seq["mixup"] = MixUpAugment(
                min_mix=self.params.min_mix,
                max_mix=self.params.max_mix
                )
        # Legacy condition. To update.
        if self.params.cutmix or self.params.n_cutmix >= 1:
            self.tf_seq["cutmix"] = CutMixAugment(
                min_mix=self.params.min_mix,
                max_mix=self.params.max_mix
                )

    def init_tag(self):
        """How to save exp name. ex : m200mbs100sbs10
        """
        if self.params.training_type == 'inc':
            self.params.tag += f"_m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}"
        else:
            self.params.tag += f"_e{self.params.epochs}b{self.params.batch_size},uni"
        lg.info(f"Using the following tag for this experiment : {self.params.tag}")

    def load_criterion(self):
        return SupConLoss(0.07)

    def train(self, dataloader, **kwargs):
        self.model = self.model.train()
        if self.params.training_type == 'inc':
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "uni":
            self.train_uni(dataloader, **kwargs)

    def train_uni(self, dataloader, **kwargs):
        task_name = kwargs.get("epoch")
        for j, batch in enumerate(dataloader):
            batch_x, batch_y = batch[0].to(self.device), batch[1].to(self.device)
            
            augmentations = self.augment(combined_x=batch_x, mem_x=batch_x, batch_x=batch_x)
            proj_list = []
            for aug in augmentations:
                _, p = self.model(aug)
                proj_list.append(p.unsqueeze(1))

            projections = torch.cat(proj_list, dim=1)
            # Loss
            loss = self.criterion(
                features=projections,
                # proj_idx=combined_idx,
                labels=batch_y if self.params.supervised else None,
                )
            loss = loss.mean()
            self.loss = loss.item()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
    
            # Plot to tensorboard
            if self.params.tensorboard:
                self.plot()

            if ((not j % self.params.test_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
                self.save(model_name=f"ckpt_{task_name}.pth")


    def train_inc(self, dataloader, **kwargs):
        task_name = kwargs.get("task_name")

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y, batch_idx = batch[0], batch[1], batch[2]
            self.stream_idx += len(batch_x)

            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Combined batch
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)

                    # Augment
                    augmentations = self.augment(combined_x=combined_x, mem_x=mem_x.to(self.device), batch_x=batch_x.to(self.device), task_id=kwargs.get('task_id', 0))

                    self.model.train()

                    # Inference
                    proj_list = []
                    for aug in augmentations:
                        _, p = self.model(aug)
                        proj_list.append(p.unsqueeze(1))
                    
                    projections = torch.cat(proj_list, dim=1)

                    # Loss
                    loss = self.criterion(
                        features=projections,
                        # proj_idx=combined_idx,
                        labels=combined_y if self.params.supervised else None,
                        )
                    loss = loss.mean()
                    self.loss = loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()                

            # Update buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, idx_data=batch_idx, model=self.model)

            # Plot to tensorboard
            if self.params.tensorboard:
                self.plot()

            if ((not j % self.params.test_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
                self.save(model_name=f"ckpt_{task_name}.pth")

    def augment(self, combined_x, mem_x, batch_x, **kwargs):
        with torch.no_grad():
            augmentations = []
            for key in self.tf_seq:
                if 'aug' in key:
                    augmentations.append(self.tf_seq[key](combined_x))
                else:
                    n_repeat = self.get_n_repeat(key)
                    batch1 = combined_x
                    batch2 = batch_x
                    for _ in range(n_repeat):
                        augmentations.append(self.tf_seq[key](batch1, batch2, model=self.model))
            return augmentations
    
    def get_n_repeat(self, augmentation):
        if augmentation == 'style':
            return self.params.n_styles
        if augmentation == 'cutmix':
            return self.params.n_cutmix
        if augmentation == 'mixup':
            return self.params.n_mixup
        return 1
    
    def after_train(self, **kwargs):
        if self.params.jfmix:
            self.tf_seq['jfmix'].save()
    
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

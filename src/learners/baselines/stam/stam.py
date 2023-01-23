"""Adapted from https://github.com/CameronTaylorFL/stam
"""
import torch
import time
import torch.nn as nn
import sys
import numpy as np
import os
import random as r
import pickle
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
import ctypes
from functools import partial

import config.stam_config as config

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

from src.learners.base import BaseLearner
from src.learners.baselines.stam.STAM_classRepo import Layer
from src.utils.metrics import forgetting_line   

class STAMLearner(BaseLearner):
    def __init__(self, args):
        if args.dataset == 'cifar10':
            self.config = config.cfg_cifar10
        elif args.dataset == 'cifar100':
            self.config = config.cfg_cifar100
        elif args.dataset == 'tiny':
            self.config = config.cfg_tiny
        super().__init__(args)
        self.classifiers_list = ['topdown']
        self.init_results()
        self.images_seen = 0
        
    def load_optim(self):
        pass
    
    def load_criterion(self):
        pass

    def load_model(self):
        ###############################################################################################
        # layers - [name, rf, stride, stride-reconstruct, num_cents, alpha-ltm, alpha-stm, beta,theta #
        ###############################################################################################

        # load mnist/emnist architecture
        if self.params.dataset == 'mnist' or self.params.dataset == 'emnist':
            self.config['num_phases'] =self.params.n_tasks
            if self.config["model_flag"] == 1:
                self.config['layers'] = [['L1', 8,  1, 2, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]],
                        ['L2', 13, 1, 3, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]],
                        ['L3', 20, 1, 4, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]]]

        # load cifar/svhn architecture
        if self.params.dataset == 'svhn':  
            self.config['num_phases'] =self.params.n_tasks
            if self.config["model_flag"] == 1:
                self.config['layers'] = [['L1', 10, 2, 2, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]],
                        ['L2', 14, 2, 3, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]],
                        ['L3', 18, 2, 2, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]]]
        
        if self.params.dataset == 'cifar10':  
            self.config['num_phases'] =self.params.n_tasks
            if self.config["model_flag"] == 1:
                self.config['layers'] = [['L1', 12, 2, 2, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]],
                        ['L2', 18, 2, 3, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]],
                        ['L3', 22, 2, 2, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]]]
        
        if self.params.dataset == 'cifar100' or self.params.dataset == 'imagenet':
            self.config['num_phases'] = self.params.n_tasks
            if self.config["model_flag"] == 1:
                self.config['layers'] = [['L1', 10, 2, 2, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]],
                        ['L2', 14, 2, 3, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]],
                        ['L3', 18, 2, 2, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]]]

        if self.params.dataset == 'tiny':
            self.config['num_phases'] = self.params.n_tasks
            if self.config["model_flag"] == 1:
                self.config['layers'] = [['L1', 10, 2, 1, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]],
                        ['L2', 18, 2, 2, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]],
                        ['L3', 26, 2, 2, self.config["delta"], self.config["alpha"], 0.0, self.config["beta"], self.config["theta"]]]
                      
        # layers to perform classification               
        if self.config["layers_flag"] == 0:
            l_eval = []
            l_eval_name = []
        elif self.config["layers_flag"] == 1:
            l_eval = [l for l in range(len(self.config['layers']))]
            l_eval_name = ['all']
        elif self.config["layers_flag"] == 2:
            l_eval = [0, 1]
            l_eval_name = ['two']
        elif self.config["layers_flag"] == 3:
            l_eval = [0]
            l_eval_name = ['one']

        self.config['l_eval'] = l_eval
        self.config['l_eval_name'] = l_eval_name
        self.config['expected_features'] = [200 for _ in range(len(self.config['layers']))]

        self.layers = []
        self.layers.append(Layer(self.params.img_size, self.params.nb_channels, *self.config['layers'][0], 
                                self.config['WTA'], self.config["im_scale"], self.config["scale_flag"], 
                                self.params.seed, self.config['kernel'], self.config["expected_features"][0],
                                self.config["nd_fixed"], self.config["init_size"], os.path.join('results/', self.params.tag), self.config['vis']))
        for l in range(1,len(self.config['layers'])):
            self.layers.append(Layer(self.params.img_size, self.params.nb_channels, *self.config['layers'][l],
                                    self.config['WTA'], self.config["im_scale"], self.config["scale_flag"], 
                                    self.params.seed, self.config['kernel'],  self.config["expected_features"][l],
                                    self.config["nd_fixed"], self.config["init_size"], self.params.tag, self.config['vis']))
    
        self.init_layers()

        # classification parameters
        self.Fz = [[] for i in range(self.config['num_samples'])]
        self.D = [[] for i in range(self.config['num_samples'])]
        self.D_sum = [[] for i in range(self.config['num_samples'])]
        self.cent_g = [[] for i in range(self.config['num_samples'])]
        self.Nl_seen = [0 for i in range(self.config['num_samples'])]

        # visualize task boundaries
        self.ndy = []

    def init_layers(self):
        np.random.seed(self.params.seed)
        # for all layers
        for l in range(len(self.layers)):
        
            # number of centroids to initialize
            n_l = self.layers[l].num_cents
                                                        
            # random init
            self.layers[l].centroids = np.random.randn(n_l, 
                                                       self.layers[l].recField_size \
                                                       * self.layers[l].recField_size \
                                                       * self.layers[l].ch) * 0.1 

            # normalize sum to 1
            self.layers[l].centroids -= np.amin(self.layers[l].centroids, axis = 1)[:,None]
            self.layers[l].centroids /= np.sum(self.layers[l].centroids, axis = 1)[:,None]

    def init_tag(self):
        """How to save exp name. ex : m200mbs100sbs10
        """
        self.params.tag += f""
        print(f"Using the following tag for this experiment : {self.params.tag}")

    def train(self, dataloader, task_name, **kwargs):
        # reset d samples - for visualization
        for l in range(len(self.layers)):
            self.layers[l].d_sample = 100
            self.layers[l].delete_n = []
        self.ndy.append(self.images_seen+1)

        # Iterate over data
        for j, batch in enumerate(dataloader):
            # Convert to numpy format (probably not optimal) TODO: update this terrible code
            batch_x = np.ascontiguousarray(np.swapaxes(np.swapaxes(np.array(batch[0].squeeze(0)), 0, 1), 1, 2)*255)
            batch_y = np.array(batch[1].squeeze(0))
            
            # reset d samples
            if j == len(dataloader) - 100:
                for l in range(len(self.layers)):
                    self.layers[l].d_sample = 100
            
            self.images_seen = self.images_seen + 1
            self.train_update(batch_x, batch_y)

            if ((not j % self.params.test_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                print(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)} time : {time.time() - self.start:.4f}s",
                    end="\r"
                )

    def train_update(self, x, y):
        # for all layers
        x_i = x
        for l in range(len(self.layers)):    
            x_i = self.layers[l].forward(x_i, y, update=True)  
            
    def supervise(self, data, labels, phase, index=0, l_list=None, image_ret=False, vis=False):
        # process inputs
        num_data = len(data)

        # get centroids for classification
        self.cents_ltm = []
        self.class_ltm = []

        for l in range(len(self.layers)):
            if self.layers[l].num_ltm > 0:
                self.cents_ltm.append(self.layers[l].get_ltm_centroids())
                self.class_ltm.append(self.layers[l].get_ltm_classes())
            else:
                self.cents_ltm.append(self.layers[l].get_stm_centroids())
                self.class_ltm.append(self.layers[l].get_stm_classes())

        # this is repeat of self.setTask which is kept for scenario
        # where labeled data is NOT replayed
        if self.Nl_seen[index] == 0:
            self.D_sum[index] = [0 for l in range(len(self.config['l_eval']))]
            self.D[index] = [[] for l in range(len(self.config['l_eval']))]
            self.Fz[index] = [[] for l in range(len(self.config['l_eval']))]
            self.cent_g[index] = [[] for l in range(len(self.config['l_eval']))]    
        self.Nl_seen[index] += num_data

        # supervision per layer
        for l_index in range(len(self.config['l_eval'])):

            # get layer index from list of classification layers
            l = self.config['l_eval'][l_index]
        
            # get layer centroids
            centroids = self.cents_ltm[l]
            num_centroids = int(len(centroids))
            
            # get value of D for task
            # we use D to normalize distances wrt average centroid-patch distance
            for i in range(num_data):
            
                # get input to layer l
                x_i = data[i]
                for l_ in range(l):
                    x_i = self.layers[l_].forward(x_i, None, update = False)
            
                # extract patches
                patches = self.layers[l].extract_patches(x_i)
                shape = patches.shape
                xp = patches.reshape(self.layers[l].num_RFs, -1)
                [xp, _, _] = self.layers[l].scale(xp)
                
                # calculate and save distance
                cp = centroids.reshape(num_centroids, -1)
                d = smart_dist(xp, cp)
                close_ind = np.argmin(d, axis = 1)
                self.D_sum[index][l_index] += np.sum(d[range(shape[0]),close_ind]) / shape[0]

            # final D calculation    
            self.D[index][l_index] = self.D_sum[index][l_index] / self.Nl_seen[index]
                       
            # this holds sum of exponential "score" for each centroid for each class
            sum_fz_pool = np.zeros((num_centroids, self.params.n_classes))


            # this code is relevant if we are not replaying labeled data
            ncents_past = len(self.Fz[index][l_index])
            if ncents_past > 0:
                sum_fz_pool[:ncents_past,:] = self.Fz[index][l_index]

            # for each image
            for i in range(num_data):
            
                # get input to layer l
                x_i = data[i]
                for l_ in range(l):
                    x_i = self.layers[l_].forward(x_i, None, update = False)
            
                # extract patches
                patches = self.layers[l].extract_patches(x_i)
                shape = patches.shape
                xp = patches.reshape(self.layers[l].num_RFs, -1)
                [xp, _, _] = self.layers[l].scale(xp)
                
                # calculate distance
                cp = centroids.reshape(num_centroids, -1)
                d = smart_dist(xp, cp)

                # get distance of *matched* centroid of each patch
                close_ind = np.argmin(d, axis = 1)
                dist = (d[range(shape[0]),close_ind])

                # get exponential distance and put into sparse array with same shape as 
                # summed exponential scores if we have two centroid matches in same 
                # image, only save best match
                td = np.zeros(d.shape)
                td[range(shape[0]),close_ind] = np.exp(-1*dist/self.D[index][l_index])
                fz = np.amax(td, axis = 0)
                
                # update sum of exponential "score" for each centroid for each class
                sum_fz_pool[:, int(labels[i])] += fz

            # save data scores and calculate g values as exponential "score" normalized 
            # accross classes (i.e. score of each centroid sums to 1)
            self.Fz[index][l_index] = sum_fz_pool    
            self.cent_g[index][l_index] = np.copy(sum_fz_pool)

            for j in range(num_centroids):
                self.cent_g[index][l_index][j,:] = self.cent_g[index][l_index][j,:] \
                    / (np.sum(self.cent_g[index][l_index][j,:]) + 1e-5)


    def evaluate(self, dataloaders, task_id):
        for clf_name in self.classifiers_list:
            accs = {}
            accs[clf_name] = []

            self.setTask(self.config['num_samples'], int((self.params.n_classes / self.params.n_tasks) * (task_id + 1)))

            dl = dataloaders['train']
            step_size = int(self.params.n_classes/self.params.n_tasks)
            data, labels = self.sample_rep_label(
                dataloader=dl,
                labels_list=self.params.labels_order[:(task_id+1)*step_size],
                data_per_class=self.params.lab_pc,
                output_format="numpy",
                return_raw=True
                )
            self.supervise(data, labels, task_id)

            for j in range(task_id + 1):
                imgs_test, labels_test = self.encode(dataloaders[f"test{j}"])
                preds = self.topDownClassify(imgs_test, vis=False, phase=None)
                acc = accuracy_score(labels_test, preds)
                accs[clf_name].append(acc)
            
            for clf_name in accs:
                for _ in range(self.params.n_tasks - task_id - 1):
                    accs[clf_name].append(np.nan)
                self.results[clf_name].append(accs[clf_name])
            
            for clf_name in self.results:
                line = forgetting_line(pd.DataFrame(self.results[clf_name]), task_id=task_id)
                line = line[0].to_numpy().tolist()
                self.results_forgetting[clf_name].append(line)

        self.print_results(task_id)

        return np.nanmean(self.results['topdown'][-1]), np.nanmean(self.results_forgetting['topdown'][-1])

    # stam primary classification function - hierarchical voting mechanism
    def topDownClassify(self, test_data, index=0, experiment_params=(), vis=False, phase=None):
        # process inputs and init return labels
        num_data = len(test_data)
        labels = -1 * np.ones((num_data,))

        # for each data
        for i, data in enumerate(test_data):
            # get NN centroid for each patch
            close_ind = []
            close_distances = []
            for l in range(len(self.layers)):

                # get ltm centroids at layer
                centroids = self.cents_ltm[l]
                num_centroids = int(len(centroids))

                # get input to layer
                x_i = data
                for l_ in range(l): x_i = self.layers[l_].forward(x_i, None, update = False)

                # extract patches
                patches = self.layers[l].extract_patches(x_i)
                xp = patches.reshape(self.layers[l].num_RFs, -1)
                [xp, shifts, scales] = self.layers[l].scale(xp)

                # calculate distance
                cp = centroids.reshape(num_centroids, -1)
                d = smart_dist(xp, cp)
                close_ind.append(np.argmin(d, axis = 1))
                close_distances.append(np.min(d, axis = 1))
            
            
            # get highest layer containing at least one CIN centroid
            l = len(self.layers)-1
            found_cin = False
            while l > 0 and not found_cin:
            
                # is there at least one CIN centroid?
                if np.amax(self.cent_g[index][l][close_ind[l]]) >= self.rho_task:
                    found_cin = True
                else:
                    l -= 1
            l_cin = l
                        
            # classification
            #
            # vote of each class for all layers
            wta_total = np.zeros((self.params.n_classes,)) + 1e-3

            # for all cin layers
            layer_range = range(l_cin+1)
            percent_inform = []
            for l in layer_range:
                # vote of each class in this layer
                wta = np.zeros((self.params.n_classes,))

                # get max g value for matched centroids
                votes_g = np.amax(self.cent_g[index][l][close_ind[l]], axis = 1)

                # nullify vote of non-cin centroids
                votes_g[votes_g < self.rho_task] = 0

                
                a = np.where(votes_g > self.rho_task)
                percent_inform.append(len(a[0])/ len(votes_g))      

                # calculate per class vote at this layer
                votes = np.argmax(self.cent_g[index][l][close_ind[l]], axis = 1)
                for k in range(self.params.n_classes):
                    wta[k] = np.sum(votes_g[votes == k])

                # add to cumalitive total and normalize
                wta /= len(close_ind[l])
                
                wta_total += wta

            # final step
            labels[i] = np.argmax(wta_total)
                 
        return labels
    
    def encode(self, dataloader, nbatches=-1):
        i = 0
        with torch.no_grad():
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                imgs = np.ascontiguousarray(np.swapaxes(np.swapaxes(sample[0].cpu().numpy(), 1, 2), 2, 3)*255)
                labels = np.array(sample[1].cpu().numpy())
                
                if i == 0:
                    all_imgs = imgs
                    all_labels = labels
                else:
                    all_imgs = np.vstack([all_imgs, imgs])
                    all_labels = np.hstack([all_labels, labels])
                i += 1
        return all_imgs, all_labels

    def setTask(self, num_samples, K):
        
        # classification parameters
        self.Fz = [[] for i in range(num_samples)]
        self.D = [[] for i in range(num_samples)]
        self.D_sum = [[] for i in range(num_samples)]
        self.cent_g = [[] for i in range(num_samples)]
        self.Nl_seen = [0 for i in range(num_samples)]

        # set rho
        self.rho_task = self.config['rho']+(1/K)

def l2_dist(x, y):
    
    xx = np.sum(x**2, axis = 1)
    yy = np.sum(y**2, axis = 1)
    xy = np.dot(x, y.transpose((1,0)))

    d = xx[:,None] - 2*xy + yy[None,:]
    d[d<0] = 0
    d = (d)**(1/2)
    
    return d

def l1_dist(x, y):
    return np.sum(np.absolute(x[:,None,:]-y[None,:,:]), axis = 2)


def earth_mover_dist(x, y):
    return pairwise_distances(x, y, metric=wasserstein_distance)

def smart_dist(x, y, method="L2"):
    if method == "L1":
        return l1_dist(x, y)
    elif method == "L2":
        return l2_dist(x, y)
    elif method == "EM":
        return earth_mover_dist(x, y)
    else:
        return x
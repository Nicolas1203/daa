import os
import torch
import logging as lg
import time
import numpy as np



def get_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    return torch.device(dev)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model_weights, model_name, dir="./checkpoints/"):
    """Save PyTorch model weights
    Args:
        model_weights (Dict): model stat_dict
        model_name (str): name_of_the_model.pth
    """
    lg.debug("Saving checkpoint...")
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    torch.save(model_weights, os.path.join(dir, model_name))


# @utils.tensorfy(0, 1, tensor_klass=torch.LongTensor)
def filter_labels(y, labels):
    """Utility used to create a mask to filter values in a tensor.

    Args:
        y (list, torch.Tensor): tensor where each element is a numeric integer
            representing a label.
        labels (list, torch.Tensor): filter used to generate the mask. For each
            value in ``y`` its mask will be "1" if its value is in ``labels``,
            "0" otherwise".

    Shape:
        y: can have any shape. Usually will be :math:`(N, S)` or :math:`(S)`,
            containing `batch X samples` or just a list of `samples`.
        labels: a flatten list, or a 1D LongTensor.

    Returns:
        mask (torch.ByteTensor): a binary mask, with "1" with the respective value from ``y`` is
        in the ``labels`` filter.

    Example::

        >>> a = torch.LongTensor([[1,2,3],[1,1,2],[3,5,1]])
        >>> a
         1  2  3
         1  1  2
         3  5  1
        [torch.LongTensor of size 3x3]
        >>> classification.filter_labels(a, [1, 2, 5])
         1  1  0
         1  1  1
         0  1  1
        [torch.ByteTensor of size 3x3]
        >>> classification.filter_labels(a, torch.LongTensor([1]))
         1  0  0
         1  1  0
         0  0  1
        [torch.ByteTensor of size 3x3]
    """
    mapping = torch.zeros(y.size()).byte()

    for label in labels:
        mapping = mapping | y.eq(label)

    return mapping


def timing(function):
    """Timing decorator for code profiling

    Args:
        function : Function to evaluate (measures time performance)
    """
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time  = time.time()
        duration = (end_time- start_time)*1000.0
        f_name = function.__name__
        lg.info("{} took {:.3f} ms".format(f_name, duration))

        return result
    return wrap


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

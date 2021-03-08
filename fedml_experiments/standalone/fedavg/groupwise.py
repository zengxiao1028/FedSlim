import torch
import numpy as np
from torch.utils.data import Sampler
import torch.nn.functional as F

class EMAverage(object):
    def __init__(self, alpha=0.9):
        self.first_update = True
        self.value = 0
        self.alpha = alpha
    def update(self, value):
        if self.first_update:
            self.value = value
            self.first_update = False
        else:
            self.value = self.alpha*self.value + (1 - self.alpha) * value
    def __str__(self):
        return '{:.6f}'.format(self.value)

class SplitSet(torch.utils.data.dataset.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        if indices is not None:
            self.indices = indices
        else:
            indices = np.arange( len(self.dataset) )
            self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def get_slice(self, start, end):
        imgs = []
        targets = []

        for i in range(start, end):
            img, target = self[i]
            imgs.append(img)
            targets.append(target)

        return torch.stack(imgs), torch.LongTensor(targets)

class GroupwiseImportanceSampler(Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Arguments:
        dataset (Dataset): The dataset that performs importance
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, dataset, replacement=False):
        self.smooth_factor = 0.5
        self.dataset = dataset
        self.replacement = replacement
        self.num_samples = len(dataset)
        #for groupwise importance sampling
        self.importance = torch.ones((len(self.dataset),))
        self.group_indicator = np.zeros((len(self.dataset),))  # initial group indicator == 0
        #for importance update
        self.cur_sample_index = 0
        self.group_index = 0
        self.last_update_iteration = -1
        self.loss_ema = EMAverage()
        self.losses_window = []
        self.group_sample_size = 4

    def update_importance(self, iteration, update_batchsize, model, device='cpu'):

        if iteration > self.last_update_iteration:
            self.group_index += 1
            self.last_update_iteration = iteration

        start_index = self.cur_sample_index
        end_index = min(self.cur_sample_index + update_batchsize, len(self.dataset) )
        data, label = self.dataset.get_slice(start_index, end_index)

        # compute sample importances
        with torch.no_grad():
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            presam_losses = F.cross_entropy(output, label, reduction='none')

        self.importance[start_index:end_index] = presam_losses

        #update ema loss
        if len(self.losses_window) < 3*update_batchsize:
            self.losses_window += presam_losses.tolist()
        else:
            self.loss_ema.update(np.mean(self.losses_window))
            self.losses_window = []

        self.group_indicator[start_index:end_index] = self.group_index

        if end_index == len(self.dataset):
            self.cur_sample_index = 0
        else:
            self.cur_sample_index = end_index


    def __iter__(self):
        counter = 0

        while True:

            group_member_location = self.group_indicator==self.group_index
            group_importances = self.importance[group_member_location]
            group_importances = group_importances + self.smooth_factor * self.loss_ema.value
            group_importances = group_importances / torch.sum(group_importances)

            # this is just the group index, need to convert back to global index
            index_list = torch.multinomial(torch.Tensor(group_importances), self.group_sample_size,
                                           self.replacement).tolist()
            group_member_index = group_member_location.nonzero()[0]
            important_index = group_member_index[np.array(index_list)]

            for i in important_index:
                yield i
                counter += 1
                if counter >= self.num_samples:
                    return

    def __len__(self):
        return self.num_samples
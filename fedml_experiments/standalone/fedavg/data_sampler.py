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

class DataImportanceSampler(Sampler):

    def __init__(self, dataset, replacement=False):

        self.dataset = dataset
        self.replacement = replacement
        self.num_samples = len(dataset)
        #for groupwise importance sampling
        self.importance = torch.ones((len(self.dataset),))

        #for importance update

        self.loss_ema = EMAverage()


    def update_importance(self, model, update_batchsize=32, device='cpu', smooth_factor=0.5):
        cur_sample_index = 0
        sample_losses = torch.zeros((len(self.dataset),))
        while True:
            start_index = cur_sample_index
            end_index = min(cur_sample_index + update_batchsize, len(self.dataset) )
            data, label = self.dataset.get_slice(start_index, end_index)

            # compute sample importances
            with torch.no_grad():
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                presam_losses = F.cross_entropy(output, label, reduction='none')

            sample_losses[start_index:end_index] = presam_losses

            if end_index == len(self.dataset):
                break
            else:
                cur_sample_index = end_index

        avg_loss = torch.mean(sample_losses)
        sample_losses = sample_losses + smooth_factor * avg_loss
        self.importance[:] = sample_losses / torch.sum(sample_losses)


    def __iter__(self):
        counter = 0
        random_sample_num = min(self.num_samples, 16)
        while True:

            # this is just the group index, need to convert back to global index
            index_list = torch.multinomial(self.importance, random_sample_num, self.replacement).tolist()

            for i in index_list:
                yield i
                counter += 1
                if counter >= self.num_samples:
                    return

    def __len__(self):
        return self.num_samples
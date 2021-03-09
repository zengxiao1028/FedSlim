import torch
import torch.nn as nn
import copy
import time
import numpy as np
import logging
class CNN_OriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_OriginalFedAvg, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        return x




class CNN_DropOut(torch.nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.max_pooling(x)
        #x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        #x = self.dropout_2(x)
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        return x


class CNN_Slimmable(torch.nn.Module):
    def __init__(self, only_digits=True):
        super(CNN_Slimmable, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = USConv2d(1, 64, kernel_size=3, us=[False, True])
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = USConv2d(64, 128, kernel_size=3, us=[True, True])
        self.flatten = nn.Flatten()
        self.linear_1 = USLinear(18432, 256, us=[True,True])
        self.linear_2 = USLinear(256, 10 if only_digits else 62, us=[True, False])
        self.relu = nn.ReLU()
        self.conv2d_2.pre_layer = self.conv2d_1
        self.linear_1.pre_layer = self.conv2d_2
        self.linear_2.pre_layer = self.linear_1
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)

        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.linear_1(x)

        x = self.relu(x)
        x = self.linear_2(x)

        return x

    def set_width(self, width_mult):
        for name, module in self.named_children():
            if isinstance(module, (USConv2d, USLinear)):
                module.width_mult = width_mult

    def slim(self, slim_channels='leftmost', slim_group=0):
        for name, module in self.named_children():
            if isinstance(module, (USConv2d, USLinear)):
                module.slim(slim_channels, slim_group=slim_group)

    def trimmed_weights(self):
        w = copy.deepcopy(self.cpu().state_dict())
        for name, module in self.named_children():
            if isinstance(module, USConv2d):
                w[name+'.weight'][module.out_channels:, module.in_channels:, :, :] = torch.tensor(float('nan'))
                if module.bias is not None:
                    w[name+'.bias'][module.out_channels:] = torch.tensor(float('nan'))
            elif isinstance(module, USLinear):
                w[name+'.weight'][module.out_features:, module.in_features:] = torch.tensor(float('nan'))
                if module.bias is not None:
                    w[name+'.bias'][module.out_features:] = torch.tensor(float('nan'))
        return w

class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1, depthwise=False, bias=True, width_mult=1.0,
                 us=[True, True]):
        super(USConv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation , groups=groups, bias=bias)
        self.depthwise = depthwise
        self.width_mult = width_mult
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.us = us
        self.pre_layer = None

    def slim(self, slim_channels='leftmost', slim_group=0):
        if slim_channels=='leftmost':
            if self.us[0]:
                in_channel_num = make_divisible(self.in_channels_max * self.width_mult)
            else:
                in_channel_num = self.in_channels_max
            self.in_channels_index = torch.tensor([i for i in range(in_channel_num)])
            if self.us[1]:
                out_channel_num = make_divisible(self.out_channels_max * self.width_mult)
            else:
                out_channel_num = self.out_channels_max
            self.out_channels_index = torch.tensor([ i for i in range(out_channel_num)])
        elif slim_channels == 'random':
            if self.us[0]:
                if isinstance(self.pre_layer, USConv2d):
                    self.in_channels_index = torch.tensor(self.pre_layer.out_channels_index)
                else:
                    raise ValueError('Pre layer should be USConv2d')
            else:
                self.in_channels_index = torch.tensor([i for i in range(self.in_channels_max)])
            if self.us[1]:
                out_channel_num = make_divisible(self.out_channels_max * self.width_mult)
                self.out_channels_index = torch.tensor(np.sort(np.random.choice(self.out_channels_max, out_channel_num, False)))
            else:
                self.out_channels_index = torch.tensor([i for i in range(self.out_channels_max)])
        elif slim_channels == 'random_group':
            if self.us[0]:
                if isinstance(self.pre_layer, USConv2d):
                    self.in_channels_index = torch.tensor(self.pre_layer.out_channels_index)
                else:
                    raise ValueError('Pre layer should be USConv2d')
            else:
                self.in_channels_index = torch.tensor([i for i in range(self.in_channels_max)])
            if self.us[1]:
                out_channel_num = make_divisible(self.out_channels_max * self.width_mult)
                start = np.random.randint(0, self.out_channels_max - out_channel_num + 1)
                self.out_channels_index = torch.tensor([i for i in range(start, start + out_channel_num)])
            else:
                self.out_channels_index = torch.tensor([i for i in range(self.out_channels_max)])
        elif slim_channels == 'random_fixgroup':
            if self.us[0]:
                if isinstance(self.pre_layer, USConv2d):
                    self.in_channels_index = torch.tensor(self.pre_layer.out_channels_index)
                else:
                    raise ValueError('Pre layer should be USConv2d')
            else:
                self.in_channels_index = torch.tensor([i for i in range(self.in_channels_max)])
            if self.us[1]:
                if self.width_mult == 1.0:
                    self.out_channels_index = torch.tensor([i for i in range(self.out_channels_max)])
                elif self.width_mult == 0.5:
                    out_channel_num = make_divisible(self.out_channels_max * self.width_mult)
                    if slim_group==0:
                        self.out_channels_index = torch.tensor([i for i in range(0, out_channel_num)])
                    else:
                        self.out_channels_index = torch.tensor([i for i in range(out_channel_num, self.out_channels_max)])
                else:
                    raise ValueError('random_fixgroup width should be 1.0 or 0.5')
            else:
                self.out_channels_index = torch.tensor([i for i in range(self.out_channels_max)])
        else:
            raise NotImplementedError()

    def forward(self, input):

        self.groups = self.in_channels if self.depthwise else 1

        #weight = self.weight[:len(self.out_channels_index), :len(self.in_channels_index), :, :]
        weight = self.weight[self.out_channels_index, :, :, :][:, self.in_channels_index, :, :]

        if self.bias is not None:
           bias = self.bias[self.out_channels_index]
        else:
           bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, width_mult=1.0, us=[True, True]):
        super(USLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.us = us
        self.width_mult = width_mult
        self.pre_layer = None


    def slim(self, slim_channels='leftmost', slim_group=0):
        if slim_channels=='leftmost':
            if self.us[0]:
                in_feature_num = make_divisible(self.in_features_max * self.width_mult)
            else:
                in_feature_num = self.in_features_max
            self.in_features_index = torch.tensor([i for i in range(in_feature_num)])
            if self.us[1]:
                out_feature_num = make_divisible(self.out_features_max * self.width_mult)
            else:
                out_feature_num = self.out_features_max
            self.out_features_index = torch.tensor([ i for i in range(out_feature_num)])

        elif slim_channels == 'random':
            if self.us[0]:
                if isinstance(self.pre_layer, USConv2d):
                    all_indexes = np.arange(self.in_features_max).reshape(self.pre_layer.out_channels_max, -1)
                    all_indexes = all_indexes[self.pre_layer.out_channels_index, :].flatten()
                    self.in_features_index = torch.tensor(all_indexes)
                elif isinstance(self.pre_layer, USLinear):
                    self.in_features_index = torch.tensor(self.pre_layer.out_features_index)
                else:
                    raise ValueError('Pre layer should be USLinear or USConv2d')
            else:
                self.in_features_index = torch.tensor([i for i in range(self.in_features_max)])
            if self.us[1]:
                out_feature_num = make_divisible(self.out_features_max * self.width_mult)
                self.out_features_index = torch.tensor(np.sort(np.random.choice(self.out_features_max, out_feature_num, False)))
            else:
                self.out_features_index = torch.tensor([i for i in range(self.out_features_max)])
        elif slim_channels == 'random_fixgroup':
            if self.us[0]:
                if isinstance(self.pre_layer, USConv2d):
                    all_indexes = np.arange(self.in_features_max).reshape(self.pre_layer.out_channels_max, -1)
                    all_indexes = all_indexes[self.pre_layer.out_channels_index, :].flatten()
                    self.in_features_index = torch.tensor(all_indexes)
                elif isinstance(self.pre_layer, USLinear):
                    self.in_features_index = torch.tensor(self.pre_layer.out_features_index)
                else:
                    raise ValueError('Pre layer should be USLinear or USConv2d')
            else:
                self.in_features_index = torch.tensor([i for i in range(self.in_features_max)])
            if self.us[1]:
                if self.width_mult == 1.0:
                    self.out_features_index = torch.tensor([i for i in range(self.out_features_max)])
                elif self.width_mult == 0.5:
                    out_feature_num = make_divisible(self.out_features_max * self.width_mult)
                    if slim_group == 0 :
                        self.out_features_index = torch.tensor([i for i in range(0, out_feature_num)])
                    else:
                        self.out_features_index = torch.tensor(
                            [i for i in range(out_feature_num, self.out_features_max)])
                else:
                    raise ValueError('random_fixgroup width should be 1.0 or 0.5')
            else:
                self.out_features_index = torch.tensor([i for i in range(self.out_features_max)])
        elif slim_channels == 'random_group':
            if self.us[0]:
                if isinstance(self.pre_layer, USConv2d):
                    all_indexes = np.arange(self.in_features_max).reshape(self.pre_layer.out_channels_max, -1)
                    all_indexes = all_indexes[self.pre_layer.out_channels_index, :].flatten()
                    self.in_features_index = torch.tensor(all_indexes)
                elif isinstance(self.pre_layer, USLinear):
                    self.in_features_index = torch.tensor(self.pre_layer.out_features_index)
                else:
                    raise ValueError('Pre layer should be USLinear or USConv2d')
            else:
                self.in_features_index = torch.tensor([i for i in range(self.in_features_max)])
            if self.us[1]:
                out_feature_num = make_divisible(self.out_features_max * self.width_mult)
                start = np.random.randint(0, self.out_features_max - out_feature_num + 1)
                self.out_features_index = torch.tensor(
                    [i for i in range(start, start + out_feature_num)])
            else:
                self.out_features_index = torch.tensor([i for i in range(self.out_features_max)])
        else:
            raise NotImplementedError()



    def forward(self, input):
        # weight = self.weight[:len(self.out_features_index), :len(self.in_features_index)]
        weight = self.weight[self.out_features_index, :][:,  self.in_features_index]
        if self.bias is not None:
            bias = self.bias[self.out_features_index]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
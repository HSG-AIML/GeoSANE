# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import enable_grad
import numpy as np
import torch.nn.functional as F

from torchvision.transforms import v2 as T



from sklearn.metrics import average_precision_score

from timm import create_model
from timm.loss import SoftTargetCrossEntropy
from shrp.models.modules.mixup_cutmix_randerase_aug import get_data_augmentations
from shrp.models.modules.info_nce_loss import InfoNCELoss
from transformers import BertModel

import timeit

import logging

from shrp.models.def_resnet_width import ResNet_width

"""
define net
##############################################################################
"""


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, nlin="relu"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            self.get_nonlin(nlin),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.get_nonlin(nlin),
        )

    def get_nonlin(self, nlin):
        return {
            "relu": nn.ReLU(inplace=True),
            "leakyrelu": nn.LeakyReLU(inplace=True),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
        }[nlin]

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, nlin="relu"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, nlin=nlin)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, nlin="relu"):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, nlin)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, nlin=nlin)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=8, bilinear=True, nlin="relu", dropout=0.0, init_type="kaiming_uniform"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.nlin = nlin
        self.dropout = dropout

        self.inc = DoubleConv(n_channels, 64, nlin=nlin)
        self.down1 = Down(64, 128, nlin=nlin)
        self.down2 = Down(128, 256, nlin=nlin)
        self.down3 = Down(256, 512, nlin=nlin)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, nlin=nlin)
        self.up1 = Up(1024, 512 // factor, bilinear, nlin=nlin)
        self.up2 = Up(512, 256 // factor, bilinear, nlin=nlin)
        self.up3 = Up(256, 128 // factor, bilinear, nlin=nlin)
        self.up4 = Up(128, 64, bilinear, nlin=nlin)
        self.outc = OutConv(64, n_classes)

        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        logging.info("Initializing weights with %s", init_type)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif init_type == "uniform":
                    nn.init.uniform_(m.weight)
                elif init_type == "normal":
                    nn.init.normal_(m.weight)
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight)
                elif init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class MLP(nn.Module):
    def __init__(
        self,
        i_dim=14,
        h_dim=[30, 15],
        o_dim=10,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
        use_bias=True,
    ):
        super().__init__()
        self.use_bias = use_bias
        # init module list
        self.module_list = nn.ModuleList()

        # get hidden layer's list
        # wrap h_dim in list of it's not already
        if not isinstance(h_dim, list):
            try:
                h_dim = [h_dim]
            except Exception as e:
                logging.error(e)
        # add i_dim to h_dim
        h_dim.insert(0, i_dim)

        # get if bias should be used or not
        for k in range(len(h_dim) - 1):
            # add linear layer
            self.module_list.append(
                nn.Linear(h_dim[k], h_dim[k + 1], bias=self.use_bias)
            )
            # add nonlinearity
            if nlin == "elu":
                self.module_list.append(nn.ELU())
            if nlin == "celu":
                self.module_list.append(nn.CELU())
            if nlin == "gelu":
                self.module_list.append(nn.GELU())
            if nlin == "leakyrelu":
                self.module_list.append(nn.LeakyReLU())
            if nlin == "relu":
                self.module_list.append(nn.ReLU())
            if nlin == "tanh":
                self.module_list.append(nn.Tanh())
            if nlin == "sigmoid":
                self.module_list.append(nn.Sigmoid())
            if nlin == "silu":
                self.module_list.append(nn.SiLU())
            if dropout > 0:
                self.module_list.append(nn.Dropout(dropout))
        # init output layer
        self.module_list.append(nn.Linear(h_dim[-1], o_dim, bias=self.use_bias))
        # normalize outputs between 0 and 1
        # self.module_list.append(nn.Sigmoid())

        # initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        logging.info("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                if self.use_bias:
                    m.bias.data.fill_(0.01)

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            logging.debug(f"layer {layer}")
            logging.debug(f"input shape:: {x.shape}")
            x = layer(x)
            logging.debug(f"output shape:: {x.shape}")
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            activations.append(x)
        return x, activations


###############################################################################
# define net
# ##############################################################################
def compute_outdim(i_dim, stride, kernel, padding, dilation):
    o_dim = (i_dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return o_dim


class CNN(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        logging.info("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


class CNN2(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(6, 9, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(9, 6, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 6, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        logging.info("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                activations.append(x)
        return x, activations


class CNN3(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 32x32 image size
        ## chn_in * 32 * 32
        ## compose layer 0
        self.module_list.append(nn.Conv2d(channels_in, 16, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 16 * 15 * 15
        ## compose layer 1
        self.module_list.append(nn.Conv2d(16, 32, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 32 * 7 * 7 // 32 * 6 * 6
        ## compose layer 2
        self.module_list.append(nn.Conv2d(32, 15, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 15 * 2 * 2
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(15 * 2 * 2, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        logging.info("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


################################################################################################
class ResCNN(nn.Module):
    """
    extension of the CNN class.
    Added residual connections via max-pooling and 1d convs to the conv layers
    Added a function to load state-dicts from the cnns without res cons
    """

    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()

        if dropout > 0.0:
            raise NotImplementedError(
                "dropout is not yet impemented for the residual connections.|"
            )
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        # input shape bx1x28x28

        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## output [15, 8, 12, 12]
        ## residual connection stack 1
        self.res1_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res1_conv = nn.Conv2d(
            in_channels=channels_in, out_channels=8, kernel_size=1, stride=1, padding=0
        )

        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## output [15, 6, 4, 4]
        self.res2_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res2_conv = nn.Conv2d(
            in_channels=8, out_channels=6, kernel_size=1, stride=1, padding=0
        )

        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## output [15, 4, 3, 3]
        self.res3_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.res3_conv = nn.Conv2d(
            in_channels=6, out_channels=4, kernel_size=1, stride=1, padding=0
        )

        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        logging.info("initialze model")
        for m in self.module_list:
            m = self.init_single(init_type, m)

        init_type = "kaiming_uniform"  # init the residual blocks with kaiming always
        self.res1_conv = self.init_single(init_type, self.res1_conv)
        self.res2_conv = self.init_single(init_type, self.res2_conv)
        self.res3_conv = self.init_single(init_type, self.res3_conv)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            m.bias.data.fill_(0.01)
        return m

    def get_nonlin(self, nlin):
        """
        gets nn class object for keyword
        """
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def load_weights_from_cnn_checkpoint(self, check):
        """
        takes weights and biases from CNN architecture without residual connections
        assumes this particular data structure, will not fail gracefully otherwise
        """
        self.module_list[0].weight.data = check["module_list.0.weight"]
        self.module_list[0].bias.data = check["module_list.0.bias"]
        self.module_list[3].weight.data = check["module_list.3.weight"]
        self.module_list[3].bias.data = check["module_list.3.bias"]
        self.module_list[6].weight.data = check["module_list.6.weight"]
        self.module_list[6].bias.data = check["module_list.6.bias"]
        self.module_list[9].weight.data = check["module_list.9.weight"]
        self.module_list[9].bias.data = check["module_list.9.bias"]
        self.module_list[11].weight.data = check["module_list.11.weight"]
        self.module_list[11].bias.data = check["module_list.11.bias"]

    def forward(self, x):
        # layer 1
        x_ = x.clone()
        for idx in range(0, 3):
            x_ = self.module_list[idx](x_)
        x = self.res1_pool(x)
        x = self.res1_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # layer 2
        x_ = x.clone()
        for idx in range(3, 6):
            x_ = self.module_list[idx](x_)
        x = self.res2_pool(x)
        x = self.res2_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # layer 3
        x_ = x.clone()
        for idx in range(6, 8):
            x_ = self.module_list[idx](x_)
        x = self.res3_pool(x)
        x = self.res3_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # flatten
        # fc1
        # fc2
        for m in self.module_list[8:]:
            x = m(x)

        return x


################################################################################
class CNN_more_layers(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, 8, 5),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(8, 4, 5, bias=False),
            nn.Tanh(),
            nn.Conv2d(4, 6, 4, bias=False),
            nn.Tanh(),
            nn.Conv2d(6, 4, 3, bias=False),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(36, 18),
            nn.Tanh(),
            nn.Linear(18, 10),
        )

        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        logging.info("initialze model")
        for m in self.layers:
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m

    def forward(self, x):
        return self.layers(x)


###############################################################################
class CNN_residual(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, 8, 5, bias=False)
        self.pool1 = nn.MaxPool2d(2)
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(8, 6, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.act2 = nn.Tanh()

        self.conv3 = nn.Conv2d(6, 4, 2, bias=False)
        self.act3 = nn.Tanh()

        self.identity = nn.Conv2d(8, 4, 1, bias=False)

        self.flatten = nn.Flatten()

        self.fc4 = nn.Linear(36, 20, bias=False)
        self.act4 = nn.Tanh()

        self.fc5 = nn.Linear(20, 10)

        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        logging.info("initialze model")
        self.conv1 = self.init_single(init_type, self.conv1)
        self.conv2 = self.init_single(init_type, self.conv2)
        self.conv3 = self.init_single(init_type, self.conv3)
        self.fc4 = self.init_single(init_type, self.fc4)
        self.fc5 = self.init_single(init_type, self.fc5)

        init_type = "kaiming_uniform"
        self.identity = self.init_single(init_type, self.identity)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m

    def forward(self, x):
        x = self.conv1(x)
        logging.debug(x.shape)

        x = self.act1(self.pool1(x))
        logging.debug(x.shape)

        x_identity = self.identity(x)
        logging.debug("x_identity", x_identity.shape)

        x = self.act2(self.pool2(self.conv2(x)))
        logging.debug(x.shape)

        x = self.act3(self.conv3(x))
        logging.debug(x.shape)

        x = x + x_identity[:, :, ::4, ::4]

        logging.debug("skip connection applied")

        x = self.flatten(x)

        x = self.act4(self.fc4(x))
        logging.debug(x.shape)

        x = self.fc5(x)
        logging.debug(x.shape)

        return x


###############################################################################
class CNN_more_layers_residual(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, 8, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(8, 4, 5, bias=False)
        self.act2 = nn.Tanh()

        self.conv3 = nn.Conv2d(4, 6, 4, bias=False)
        self.act3 = nn.Tanh()

        self.conv4 = nn.Conv2d(6, 4, 3, bias=False)
        self.act4 = nn.Tanh()

        self.flatten = nn.Flatten()

        self.fc5 = nn.Linear(36, 18)
        self.act5 = nn.Tanh()

        self.fc6 = nn.Linear(18, 10)

        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        logging.info("initialze model")
        self.conv1 = self.init_single(init_type, self.conv1)
        self.conv2 = self.init_single(init_type, self.conv2)
        self.conv3 = self.init_single(init_type, self.conv3)
        self.conv4 = self.init_single(init_type, self.conv4)
        self.fc5 = self.init_single(init_type, self.fc5)
        self.fc6 = self.init_single(init_type, self.fc6)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass

        return m

    def forward(self, x):
        x = self.conv1(x)
        logging.debug(x.shape)

        x = self.act1(self.pool1(x))
        logging.debug(x.shape)

        x = self.act2(self.conv2(x))
        logging.debug(x.shape)

        x_identity = x.clone()

        x = self.act3(self.conv3(x))
        logging.debug(x.shape)

        x = self.act4(self.conv4(x))
        logging.debug(x.shape)

        x = x + x_identity[:, :, 1:-1, 1:-1][:, :, ::2, ::2]

        logging.debug("skip connection applied")

        x = self.flatten(x)

        x = self.act5(self.fc5(x))
        logging.debug(x.shape)

        x = self.fc6(x)
        logging.debug(x.shape)

        return x


import torchvision

from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as ResNetBase


class ResNet(ResNetBase):
    """
    Wrapper for pytroch ResNet class to get access to forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_activations(self, x):
        """forward pass and return activations"""
        # See note [TorchScript super()]
        activations = []

        # input block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        activations.append(x.clone())

        # residual blocks
        x = self.layer1(x)
        activations.append(x.clone())
        x = self.layer2(x)
        activations.append(x.clone())
        x = self.layer3(x)
        activations.append(x.clone())
        x = self.layer4(x)
        activations.append(x.clone())

        # output block
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, activations


class ResNetBase(ResNet):
    """
    ResNet base class, defaults to ResNet 18, implements init and weight init
    """

    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    ):
        # call init from parent class
        super().__init__(block=block, layers=layers, num_classes=out_dim)
        # adpat first layer to fit dimensions
        self.conv1 = nn.Conv2d(
            channels_in,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.maxpool = nn.Identity()

        if init_type is not None:
            self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        for m in self.modules():
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m


class ResNet18(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet34(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[3, 4, 6, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet50(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 4, 6, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet101(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 4, 23, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet152(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 8, 36, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=Bottleneck,
            layers=layers,
        )


class ResNetBase_width(ResNet_width):
    """
    ResNet base class, defaults to ResNet 18, implements init and weight init
    """

    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        width=64,
    ):
        # call init from parent class
        super().__init__(block=block, layers=layers, num_classes=out_dim, width=width)
        # adpat first layer to fit dimensions
        self.conv1 = nn.Conv2d(
            channels_in,
            width,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.maxpool = nn.Identity()

        if init_type is not None:
            self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        for m in self.modules():
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m


class ResNet18_width(ResNetBase_width):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        width=64,
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
            width=width,
        )


class ResNet34_width(ResNetBase_width):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        width=64,
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
            width=width,
        )


class ResNet50_width(ResNetBase_width):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        width=64,
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
            width=width,
        )


class ResNet101_width(ResNetBase_width):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        width=64,
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
            width=width,
        )


class ResNet152_width(ResNetBase_width):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        width=64,
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=Bottleneck,
            layers=layers,
            width=width,
        )


class MiniAlexNet(nn.Module):
    def __init__(self, channels_in=3, num_classes=10, init_type="kaiming_uniform"):
        super(MiniAlexNet, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(
            channels_in, 96, kernel_size=3, stride=1
        )  # Use padding to keep size 32x32
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=3
        )  # Reduce size from 32x32 to 10x10
        self.batchnorm1 = nn.BatchNorm2d(96)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(96, 256, kernel_size=3, stride=1)  # Keep size 10x10
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Reduce size from 10x10 to 5x5
        self.batchnorm2 = nn.BatchNorm2d(256)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 384)  # Adjusted for 5x5 feature map size
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

        if init_type is not None:
            self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        for m in self.modules():
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m

    def forward(self, x):
        # Apply Convolutional Layers
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.maxpool2(x)

        x = x.view(-1, 256 * 4 * 4)

        # Apply Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

###############################################################################

class ViTSmallPatch16(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 init_type="kaiming_uniform",
                 fc_mlp=False,
                 hidden_dim=256,
                 embedding_dim=384,
                 dropout=0.,
                 attn_dropout=0.,
                 mixup=None,
                 cutmix=None,
                 random_erase=None):
        super(ViTSmallPatch16, self).__init__()
        self.vit = create_model("vit_small_patch16_224",
                                pretrained=False,
                                num_classes=num_classes,
                                proj_drop_rate=dropout,
                                attn_drop_rate=attn_dropout)

        if fc_mlp:
            self.vit.head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(), # currently only relu is supported
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.vit.head = nn.Linear(embedding_dim, num_classes)

        if init_type is not None:
            self.initialize_weights(init_type)

        if mixup is not None:
            self.mixup = mixup
        else:
            self.mixup = None

        if cutmix is not None:
            self.cutmix = cutmix
        else:
            self.cutmix = None

        if random_erase is not None:
            self.random_erase = random_erase

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        for m in self.modules():
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m

    def reset_classifier(self, num_classes=100, fc_mlp=False, hidden_dim=256):
        if fc_mlp:
            self.vit.head = nn.Sequential(
                nn.Linear(384, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.vit.head = nn.Linear(384, num_classes)

    def forward(self, x, y=None):
        # apply mixup, cutmix and random erase during training if enabled
        if self.training and y is not None: # only apply if y is supplied. hacky, but allows to train sampled models with or without mixup, cutmix and randerase
            if self.mixup is not None:
                x, y = self.mixup(x, y)
            if self.random_erase is not None:
                x = self.random_erase(x)
            x = self.vit(x)
            return x, y
        else:
            x = self.vit(x)
            return x

class SentimentClassifier(nn.Module):

    def __init__(self, n_classes, case):
        super(SentimentClassifier, self).__init__()

        ## Differentiate btw. continued and scratch (I expected both to be same, but somehow it does not match)
        if case == "continued":
            base_model = "bert-base-cased"

        elif case == "scratch":
            base_model = "bert-base-uncased"

        self.bert = BertModel.from_pretrained(base_model)

        self.classification_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes))

        # Initialize MLP layers with Kaiming (He) initialization
        for m in self.classification_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

        # Load pre-trained model
        # weights = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
        # #weights.pop("bert.embeddings.position_ids")
        # self.load_state_dict(weights) ## There is one more key in the saved .bin file, which is not in the original. The line above removes it. Alternatively, you can set strict=False in the load_state_dict method.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def reset_classifier(self, n_classes):
        self.classification_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes))

        # Initialize MLP layers with Kaiming (He) initialization
        for m in self.classification_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)


    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        return self.classification_head(pooled_output)
from shrp.models.def_models import (
    MLP,
    CNN,
    CNN2,
    CNN3,
    ResCNN,
    CNN_more_layers,
    CNN_residual,
    CNN_more_layers_residual,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet18_width,
    ResNet34_width,
    ResNet50_width,
    ResNet101_width,
    ResNet152_width,
    MiniAlexNet,
)
from shrp.models.vit import ViT


###############################################################################
# define FNNmodule
# ##############################################################################
class NNmodule(nn.Module):
    def __init__(self, config, cuda=False, seed=42, verbosity=0):
        super(NNmodule, self).__init__()
        # make sure mixup does not interfere with older model definitions
        self.use_mixup = False
        self.language_model = False
        # set verbosity
        self.verbosity = verbosity
        if not cuda:
            cuda = True if config.get("device", "cpu") == "cuda" else False
        if cuda and torch.cuda.is_available():
            self.device = "cuda"
            logging.info("cuda availabe:: use cuda")
        else:
            self.device = "cpu"
            self.cuda = False
            logging.info("cuda unavailable:: fallback to cpu")

        # setting seeds for reproducibility
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # get model architecture
        model = get_model(config)

        logging.info(f"send model to {self.device}")

        model.to(self.device)
        self.model = model

        # define loss function (criterion) and optimizer
        # set loss
        self.task = config.get("training::task", "classification")
        self.logsoftmax = False
        self.criterion_val = None
        if self.task == "classification":
            if config.get("training::loss", None) == "nll":
                self.criterion = nn.NLLLoss()
                self.logsoftmax = True
            elif config.get("training::loss", None) == "stce": # stce = soft target cross entropy
                self.criterion = SoftTargetCrossEntropy()
                # for validation we don't use soft targets
                if config.get("validation::loss", None) == "ce":
                    self.criterion_val = nn.CrossEntropyLoss()
            elif config.get("training::loss", None) == "info_nce":
                self.criterion = InfoNCELoss(
                    temperature=config.get("training::temperature", 0.07)
                )
            elif config.get("training::loss", None) == "bce":
                self.criterion = nn.BCEWithLogitsLoss()
                if config.get("validation::loss", None) == "bce":
                    self.criterion_val = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif self.task == "multilabel_classification":
            self.criterion = nn.BCEWithLogitsLoss()
            self.criterion_val = nn.BCEWithLogitsLoss()
        elif self.task == "regression":
            self.criterion = nn.MSELoss(reduction="mean")
        elif self.task == "segmentation":
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
            if config.get("training::loss", None) == "bce":
                self.criterion = nn.BCEWithLogitsLoss()
                self.criterion_val = nn.BCEWithLogitsLoss()
        if self.device == "cuda":
            self.criterion.to(self.device)

        # set opimizer
        self.set_optimizer(config)

        # automatic mixed precision
        self.use_amp = (
            True if config.get("training::precision", "full") == "amp" else False
        )
        if self.use_amp:
            print(f"++++++ USE AUTOMATIC MIXED PRECISION +++++++")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # set scheduler
        self.set_scheduler(config)

        # Augmentation module on batches
        mixup_alpha = config.get("augmentation::mixup_alpha", 1.0)
        cutmix_alpha = config.get("augmentation::cutmix_alpha", 1.0)

        transforms = []
        if config.get("augmentation::use_mixup", False):
            transforms.append(T.Mixup(alpha=mixup_alpha))
        if config.get("augmentation::use_cutmix", False):
            transforms.append(T.CutMix(alpha=cutmix_alpha))

        # Compose batch-wise transforms if any
        if transforms:
            self.transforms = T.Compose(transforms)
        else:
            self.transforms = IdentityTransform()

        self.accumulation_steps = config.get("training::accumulate_gradients_steps", 1)

    # module forward function
    def forward(self, x, target=None, attention_mask=None):
        # compute model prediction
        if attention_mask is not None:
            output = self.model(x, attention_mask)
        elif self.use_mixup and target is not None:
            output, target = self.model(x, target)
            return output, target
        else:
            output = self.model(x)
        if self.logsoftmax:
            output = F.log_softmax(output, dim=1)
        return output

    def set_criterion(self, config):
        """
        Set the loss function based on the task type.

        Args:
            config (dict): Configuration dictionary containing task and loss information.

        Returns:
            criterion (torch.nn.Module): Loss function module.
        """
        self.logsoftmax = False
        if self.task == "classification":
            if config.get("training::loss", "nll") == "nll":
                criterion = nn.NLLLoss()
                self.logsoftmax = True
            else:
                criterion = nn.CrossEntropyLoss(
                    label_smoothing=config.get("augmentation::label_smoothing", 0.0)
                )
        elif self.task == "regression":
            criterion = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        if self.device == "cuda":
            criterion.to(self.device)
        return criterion

    # set optimizer function - maybe we'll only use one of them anyways..
    def set_optimizer(self, config):
        # set train_step default
        # set optimizer
        if config["optim::optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config["optim::lr"],
                momentum=config["optim::momentum"],
                weight_decay=config["optim::wd"],
                nesterov=config.get("optim::nesterov", False),
            )
            return None
        elif config["optim::optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config["optim::lr"],
                weight_decay=config["optim::wd"],
            )
            return None
        elif config["optim::optimizer"] == "adamw":
            print('the learning rate: ', config["optim::lr"])
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config["optim::lr"],
                betas=(0.9, 0.999),
                weight_decay=config["optim::wd"],
            )
            return None
        elif config["optim::optimizer"] == "rms_prop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=config["optim::lr"],
                weight_decay=config["optim::wd"],
                momentum=config["optim::momentum"],
            )
            return None
        else:
            raise ValueError(f"optimizer {config['optim::optimizer']} not recognized")

    def set_scheduler(self, config):
        if config.get("optim::scheduler", None) == None:
            self.scheduler = None
        elif config.get("optim::scheduler", None) == "OneCycleLR":
            logging.info("use onecycleLR scheduler")
            max_lr = config["optim::lr"]
            try:
                steps_per_epoch = config["scheduler::steps_per_epoch"]
            except KeyError:
                steps_per_epoch = 1
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=max_lr,
                epochs=config["training::epochs_train"],
                steps_per_epoch=steps_per_epoch,
                pct_start=config.get("scheduler::pct_start", 0.3),
            )

    def save_model(self, epoch, perf_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.ptf")
            perf_dict["state_dict"] = self.model.state_dict()
            torch.save(perf_dict, fname)
        return None

    # def compute_mIoU(self, outputs, labels, num_classes=8):
    #     total_iou = torch.zeros(num_classes, device=outputs.device)
    #     total_seen = torch.zeros(num_classes, device=outputs.device)

    #     for i in range(len(outputs)):
    #         pred = torch.argmax(outputs[i].unsqueeze(0), dim=1).flatten()
    #         label = labels[i].flatten()

    #         for cls in range(num_classes):
    #             pred_inds = pred == cls
    #             target_inds = label == cls
    #             intersection = (pred_inds & target_inds).sum()
    #             union = (pred_inds | target_inds).sum()

    #             if union > 0:
    #                 iou = intersection.float() / union.float()
    #                 total_iou[cls] += iou
    #                 total_seen[cls] += 1

    #     valid = total_seen > 0
    #     mean_iou = (total_iou[valid] / total_seen[valid]).mean().item()
    #     return mean_iou

    # def compute_mIoU(self, outputs, labels, num_classes=8):
    #     total_inter = torch.zeros(num_classes, device=outputs.device)
    #     total_union = torch.zeros(num_classes, device=outputs.device)

    #     for i in range(len(outputs)):
    #         pred = torch.argmax(outputs[i], dim=0).flatten()
    #         label = labels[i].flatten()

    #         for cls in range(num_classes):
    #             pred_inds = pred == cls
    #             target_inds = label == cls
    #             intersection = (pred_inds & target_inds).sum()
    #             union = (pred_inds | target_inds).sum()

    #             total_inter[cls] += intersection
    #             total_union[cls] += union

    #     valid = total_union > 0
    #     iou_per_class = total_inter[valid].float() / total_union[valid].float()
    #     mean_iou = iou_per_class.mean().item()
    #     return mean_iou

    # def compute_mIoU(self, outputs, labels, num_classes=2, ignore_index=255):
    #     total_inter = torch.zeros(num_classes, device=outputs.device)
    #     total_union = torch.zeros(num_classes, device=outputs.device)

    #     for i in range(len(outputs)):
    #         pred = torch.argmax(outputs[i], dim=0).flatten()
    #         label = labels[i].flatten()

    #         valid = label != ignore_index
    #         pred = pred[valid]
    #         label = label[valid]

    #         for cls in range(num_classes):
    #             pred_inds = pred == cls
    #             target_inds = label == cls
    #             intersection = (pred_inds & target_inds).sum()
    #             union = (pred_inds | target_inds).sum()

    #             total_inter[cls] += intersection
    #             total_union[cls] += union

    #     valid_classes = total_union > 0
    #     iou_per_class = total_inter[valid_classes].float() / total_union[valid_classes].float()
    #     mean_iou = iou_per_class.mean().item()
    #     return mean_iou

    def compute_mIoU(self, outputs, labels, num_classes=2, ignore_index=255):
        total_inter = torch.zeros(num_classes, device=outputs.device)
        total_union = torch.zeros(num_classes, device=outputs.device)

        outputs = torch.sigmoid(outputs)
        for i in range(len(outputs)):
            # pred = torch.argmax(outputs[i], dim=0).flatten()
            pred = (outputs[i] > 0.5).float().flatten()
            label = labels[i].flatten()

            valid = label != ignore_index
            pred = pred[valid]
            label = label[valid]

            for cls in range(num_classes):
                pred_inds = pred == cls
                target_inds = label == cls
                intersection = (pred_inds & target_inds).sum()
                union = (pred_inds | target_inds).sum()

                total_inter[cls] += intersection
                total_union[cls] += union

        valid_classes = total_union > 0
        iou_per_class = total_inter[valid_classes].float() / total_union[valid_classes].float()
        mean_iou = iou_per_class.mean().item()
        return mean_iou

    def compute_mAP(self, outputs, targets):
        if isinstance(outputs, torch.Tensor):
            outputs = torch.sigmoid(outputs).detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        num_classes = targets.shape[1]
        aps = []
        for c in range(num_classes):
            y_true_c = targets[:, c]
            y_score_c = outputs[:, c]

            if y_true_c.sum() == 0:
                continue  # skip class with no positive examples

            ap = average_precision_score(y_true_c, y_score_c)
            aps.append(ap)

        if len(aps) == 0:
            return 0.0  # fallback: no valid classes
        return np.mean(aps)

    def compile_model(self):
        logging.info("compiling the model")
        self.model = torch.compile(self.model)  # requires PyTorch 2.0
        logging.info("compiled successfully")

    # one training step / batch
    def train_step(self, input, target, attention_mask=None, step_index=0):
        # zero grads before training steps
        self.optimizer.zero_grad()
        # compute pde residual
        if attention_mask is not None:
            output = self.forward(x=input, attention_mask=attention_mask)
        target_tmp = None # for mixup we also need the adjusted labels
        if self.use_mixup: # mixup requires targets in forward pass
            output, target_tmp = self.forward(input, target)
        else:
            input[torch.isnan(input)] = 0
            output = self.forward(input)
            if self.task == "classification" and isinstance(self.criterion, nn.BCEWithLogitsLoss):
                output = output.squeeze(-1)
                target = target.float()
        # realign target dimensions

        if self.task == "regression":
            target = target.view(output.shape)

        if self.task == "segmentation":
            # target = target.long()
            target = target.float()
            target = target.unsqueeze(1)
            # compute loss
        if target_tmp is not None:
            loss = self.criterion(output, target_tmp)
        else:
            loss = self.criterion(output, target)
        loss = loss / self.accumulation_steps

        # prop loss backwards to
        self.scaler.scale(loss).backward()
        # check for gradient accumulation after accumulation steps
        if (step_index + 1) % self.accumulation_steps == 0:
            # update parameters
            self.scaler.step(self.optimizer)
            # update scaler
            self.scaler.update()
            # zero grads after optimzier
            self.optimizer.zero_grad(set_to_none=True)
            # scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
        # Accuracy calculation (for classification tasks)
        correct = 0
        total_iou = 0
        skipped = 0
        if self.task == "classification":
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                probs = torch.sigmoid(output)
                preds = (probs >= 0.5).long()   # 0 or 1
                target = target.long()
                correct = (preds == target).sum().item()
            else:
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == target).sum().item()
        elif self.task == "multilabel_classification":
            correct = self.compute_mAP(output, target)
        elif self.task == "segmentation":
            correct_batch = 0
            torch.save(output, 'train_output.pt')
            torch.save(target, 'train_target.pt')
            for i in range(len(target)):
                probs = torch.sigmoid(output[i].unsqueeze(0))
                preds = (probs > 0.5).float()
                correct = (preds.flatten() == target[i].unsqueeze(0).flatten()).float().sum()
                total = target[i].numel()
                accuracy = correct / total
                correct = accuracy
                correct = correct.item()
                correct_batch += correct




                # preds = torch.argmax(output[i].unsqueeze(0), dim=1)
                # correct = (preds.flatten() == target[i].unsqueeze(0).flatten()).float().sum()
                # total = target[i].numel()
                # accuracy = correct / total
                # correct = accuracy
                # correct = correct.item()
                # correct_batch += correct

                # preds = torch.argmax(output[i].unsqueeze(0), dim=1)
                # valid_mask = target[i] != 255
                # valid_pixels = valid_mask.sum()
                # if valid_pixels > 0:
                #     correct = (preds.squeeze()[valid_mask] == target[i][valid_mask]).float().sum()
                #     accuracy = correct / valid_pixels
                #     correct = accuracy
                #     correct = correct.item()
                #     correct_batch += correct
                # else:
                #     skipped += 1
                #     continue

            correct = correct_batch
            total_iou += self.compute_mIoU(output, target, 2)

            return loss.item(), correct, total_iou, skipped

        return loss.item(), correct

    # one training epoch
    @enable_grad()
    def train_epoch(self, trainloader, epoch, idx_out=10):
        logging.info(f"train epoch {epoch}")
        # set model to training mode
        self.model.train()

        # print('trainloader size: ', len(trainloader))


        if self.verbosity > 2:
            printProgressBar(
                0,
                len(trainloader),
                prefix="Batch Progress:",
                suffix="Complete",
                length=50,
            )
        # init accumulated loss, accuracy
        loss_acc = 0
        correct_acc = 0
        all_mious = 0
        n_data = 0
        n_batch = 0
        total_skipped = 0
        #
        if self.verbosity > 4:
            start = timeit.default_timer()

        # enter loop over batches
        for idx, data in enumerate(trainloader):
            n_batch += 1
            # check if we have a language model
            if self.language_model:
                input = data["input_ids"]
                target = data["labels"]
                attention_mask = data["attention_mask"].to(self.device)
            else:
                input, target = data
                attention_mask = None
            # send to cuda
            input, target = input.to(self.device), target.to(self.device)

            # target = target.squeeze()

            # take one training step
            if self.verbosity > 2:
                printProgressBar(
                    idx + 1,
                    len(trainloader),
                    prefix="Batch Progress:",
                    suffix="Complete",
                    length=50,)
            if attention_mask is not None:
                loss, correct = self.train_step(x=input, target=target, attention_mask=attention_mask, step_index=idx)
            else:
                if self.task == 'segmentation':
                    loss, correct, miou, skipped = self.train_step(input, target, step_index=idx)
                    all_mious += miou
                    total_skipped += skipped
                else:
                    loss, correct = self.train_step(input, target, step_index=idx)
            # scale loss with batchsize
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)

            # logging
            if idx > 0 and idx % idx_out == 0:
                loss_running = loss_acc / n_data
                if self.task == "classification":
                    accuracy = correct_acc / n_data
                elif self.task == "multilabel_classification":
                    accuracy = correct_acc / n_batch
                elif self.task == "segmentation":
                    accuracy = correct_acc / (n_data - total_skipped)
                    mious = all_mious / n_batch
                elif self.task == "regression":
                    # use r2
                    accuracy = 1 - loss_running / self.loss_mean

                if self.task == 'segmentation':
                    logging.info(
                    f"epoch {epoch} -batch {idx}/{len(trainloader)} --- running ::: loss: {loss_running}; accuracy: {accuracy}; mIoU: {mious} " #miou
                )
                else:
                    logging.info(
                        f"epoch {epoch} -batch {idx}/{len(trainloader)} --- running ::: loss: {loss_running}; accuracy: {accuracy} "
                    )

        print('number of bacthes: ', n_batch)
        if self.verbosity > 4:
            end = timeit.default_timer()
            print(f"training time for epoch {epoch}: {end-start} seconds")


        # compute epoch running losses
        loss_running = loss_acc / n_data
        if self.task == "classification":
            accuracy = correct_acc / n_data
        elif self.task == 'multilabel_classification':
            accuracy = correct_acc / n_batch
        elif self.task == "regression":
            # use r2
            accuracy = 1 - loss_running / self.loss_mean

        if self.task == 'segmentation':
            print('getting train epoch accuracy ...')
            loss_running, accuracy, miou = self.test_epoch(trainloader, epoch)
            print('train epoch accuracy: ', accuracy)
            print('train epoch miou: ', miou)
            return loss_running, accuracy, miou
        else:
            print('getting train epoch accuracy ...')
            loss_running, accuracy = self.test_epoch(trainloader, epoch)
            print('train epoch accuracy: ', accuracy)
            return loss_running, accuracy

    # test batch
    def test_step(self, input, target, attention_mask=None):
        with torch.no_grad():
            # self.model.train()

            # with torch.cuda.amp.autocast(enabled=self.use_amp):
            # forward pass: prediction
            if attention_mask is not None:
                output = self.forward(x=input, attention_mask=attention_mask)
            else:
                input[torch.isnan(input)] = 0
                output = self.forward(input)
                if self.task == "classification" and isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    output = output.squeeze(-1)
                    target = target.float()
            # realign target dimensionss
            if self.task == "regression":
                target = target.view(output.shape)

            if self.task == "segmentation":
                # target = target.long()
                target = target.unsqueeze(1)
                target = target.float()
            # compute loss
            if self.criterion_val is not None:
                loss = self.criterion_val(output, target)
            else:
                loss = self.criterion(output, target)

        correct = 0
        total_iou = 0
        skipped = 0
        if self.task == "classification":
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                probs = torch.sigmoid(output)
                preds = (probs >= 0.5).long()   # 0 or 1
                target = target.long()
                correct = (preds == target).sum().item()
            else:
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == target).sum().item()
        elif self.task == "multilabel_classification":
            correct = self.compute_mAP(output, target)
        elif self.task == "segmentation":
            correct_batch = 0
            torch.save(output, 'test_output.pt')
            torch.save(target, 'test_target.pt')
            for i in range(len(target)):
                probs = torch.sigmoid(output[i].unsqueeze(0))
                preds = (probs > 0.5).float()
                correct = (preds.flatten() == target[i].unsqueeze(0).flatten()).float().sum()
                total = target[i].numel()
                accuracy = correct / total
                correct = accuracy
                correct = correct.item()
                correct_batch += correct


                # preds = torch.argmax(output[i].unsqueeze(0), dim=1)
                # correct = (preds.flatten() == target[i].unsqueeze(0).flatten()).float().sum()
                # total = target[i].numel()
                # accuracy = correct / total
                # correct = accuracy
                # correct = correct.item()
                # correct_batch += correct

                # preds = torch.argmax(output[i].unsqueeze(0), dim=1)
                # valid_mask = target[i] != 255
                # valid_pixels = valid_mask.sum()

                # if valid_pixels > 0:
                #     correct = (preds.squeeze()[valid_mask] == target[i][valid_mask]).float().sum()
                #     accuracy = correct / valid_pixels
                #     correct = accuracy
                #     correct = correct.item()
                #     correct_batch += correct
                # else:
                #     skipped += 1
                #     continue

            correct = correct_batch
            total_iou += self.compute_mIoU(output, target, 2)

            return loss.item(), correct, total_iou, skipped

        return loss.item(), correct

    # test epoch
    def test_epoch(self, testloader, epoch):
        logging.info(f"validate at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        # initilize counters
        loss_acc = 0
        correct_acc = 0
        all_mious = 0
        n_data = 0
        n_batch = 0
        total_skipped = 0


        #####
        # def convert_bn_to_gn(module, groups=16, eps=1e-5):
        #     for name, m in module.named_children():
        #         if isinstance(m, nn.BatchNorm2d):
        #             g = min(groups, m.num_features)
        #             setattr(module, name, nn.GroupNorm(g, m.num_features, eps=eps, affine=True))
        #         else:
        #             convert_bn_to_gn(m, groups, eps)

        # convert_bn_to_gn(self.model, groups=16, eps=1e-5)

        # self.model = self.model.to(self.device)

        # self.model = self.model.float().eval()
        # with torch.no_grad():
        #     _ = self.model(torch.zeros(1, 2, 224, 224, dtype=torch.float32).to(self.device))


        #####

        for idx, data in enumerate(testloader):
            # check if we have a language model
            if self.language_model:
                input = data["input_ids"]
                target = data["labels"]
                attention_mask = data["attention_mask"].to(self.device)
            else:
                input, target = data
                attention_mask = None

            # target = target.squeeze()

            input, target = input.to(self.device), target.to(self.device)
            # perform test step on batch.
            if attention_mask is not None:
                loss, correct = self.test_step(input, target, attention_mask)
            else:
                if self.task == 'segmentation':
                    loss, correct, miou, skipped = self.test_step(input, target)
                    all_mious += miou
                    total_skipped += skipped
                else:
                    loss, correct = self.test_step(input, target)
            # scale loss with batchsize
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)
            n_batch += 1

        # logging
        # compute epoch running losses
        loss_running = loss_acc / n_data

        if self.task == "classification":
            accuracy = correct_acc / n_data
        elif self.task == "multilabel_classification":
            accuracy = correct_acc / n_batch
        elif self.task == "segmentation":
            accuracy = correct_acc / (n_data - total_skipped)
            mious = all_mious / n_batch
        elif self.task == "regression":
            accuracy = 1 - loss_running / self.loss_mean

        if self.task == 'segmentation':
            logging.info(
            f"test ::: loss: {loss_running}; accuracy: {accuracy}; mIoU: {mious} " #miou
        )
            return loss_running, accuracy, mious #miou
        else:
            logging.info(f"test ::: loss: {loss_running}; accuracy: {accuracy}")

        return loss_running, accuracy


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def get_model(config):
    # construct model
    if config["model::type"] == "MLP":
        # calling MLP constructor
        logging.info("=> creating model MLP")
        i_dim = config["model::i_dim"]
        h_dim = config["model::h_dim"]
        o_dim = config["model::o_dim"]
        nlin = config["model::nlin"]
        dropout = config["model::dropout"]
        init_type = config["model::init_type"]
        use_bias = config["model::use_bias"]
        model = MLP(i_dim, h_dim, o_dim, nlin, dropout, init_type, use_bias)
    elif config["model::type"] == "UNet":
        # calling MLP constructor
        logging.info("=> creating model UNet")
        model = UNet(
            n_channels=config["model::channels_in"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
            n_classes=config["model::o_dim"],
        )
    elif config["model::type"] == "CNN":
        # calling MLP constructor
        logging.info("=> creating model CNN")
        model = CNN(
            channels_in=config["model::channels_in"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
        )
    elif config["model::type"] == "CNN2":
        # calling MLP constructor
        logging.info("=> creating model CNN")
        model = CNN2(
            channels_in=config["model::channels_in"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
        )
    elif config["model::type"] == "CNN3":
        # calling MLP constructor
        logging.info("=> creating model CNN")
        model = CNN3(
            channels_in=config["model::channels_in"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
        )
    elif config["model::type"] == "ResCNN":
        # calling MLP constructor
        logging.info("=> creating model CNN")
        model = ResCNN(
            channels_in=config["model::channels_in"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
        )
    elif config["model::type"] == "CNN_more_layers":
        # calling MLP constructor
        logging.info("=> creating model CNN")
        model = CNN_more_layers(
            init_type=config["model::init_type"],
            channels_in=config["model::channels_in"],
        )
    elif config["model::type"] == "CNN_residual":
        # calling MLP constructor
        logging.info("=> creating model CNN")
        model = CNN_residual(
            init_type=config["model::init_type"],
            channels_in=config["model::channels_in"],
        )
    elif config["model::type"] == "CNN_more_layers_residual":
        # calling MLP constructor
        logging.info("=> creating model CNN")
        model = CNN_more_layers_residual(
            init_type=config["model::init_type"],
            channels_in=config["model::channels_in"],
        )
    elif config["model::type"] == "Resnet18":
        # calling MLP constructor
        logging.info("=> creating Resnet18")
        model = ResNet18(
            channels_in=config["model::channels_in"],
            out_dim=config["model::o_dim"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
        )
    elif config["model::type"] == "Resnet34":
        # calling MLP constructor
        logging.info("=> creating Resnet34")
        model = ResNet34(
            channels_in=config["model::channels_in"],
            out_dim=config["model::o_dim"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
        )
    elif config["model::type"] == "Resnet50":
        logging.info("=> create resnet50")
        model = ResNet50(
            channels_in=config["model::channels_in"],
            out_dim=config["model::o_dim"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
        )
    elif config["model::type"] == "Resnet101":
        logging.info("=> create resnet101")
        model = ResNet101(
            channels_in=config["model::channels_in"],
            out_dim=config["model::o_dim"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
        )
    elif config["model::type"] == "Resnet152":
        logging.info("=> create resnet152")
        model = ResNet152(
            channels_in=config["model::channels_in"],
            out_dim=config["model::o_dim"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
        )
    elif config["model::type"] == "Resnet18_width":
        # calling MLP constructor
        logging.info(f"=> creating Resnet18 width {config['model::width']}")
        model = ResNet18_width(
            channels_in=config["model::channels_in"],
            out_dim=config["model::o_dim"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
            width=config["model::width"],
        )
    elif config["model::type"] == "Resnet34_width":
        # calling MLP constructor
        logging.info(f"=> creating Resnet34 width {config['model::width']}")
        model = ResNet34_width(
            channels_in=config["model::channels_in"],
            out_dim=config["model::o_dim"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
            width=config["model::width"],
        )
    elif config["model::type"] == "Resnet50_width":
        logging.info(f"=> create resnet50 width {config['model::width']}")
        model = ResNet50_width(
            channels_in=config["model::channels_in"],
            out_dim=config["model::o_dim"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
            width=config["model::width"],
        )
    elif config["model::type"] == "Resnet101_width":
        logging.info(f"=> create resnet101 width {config['model::width']}")
        model = ResNet101_width(
            channels_in=config["model::channels_in"],
            out_dim=config["model::o_dim"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
            width=config["model::width"],
        )
    elif config["model::type"] == "Resnet152_width":
        logging.info(f"=> create resnet152 width {config['model::width']}")
        model = ResNet152_width(
            channels_in=config["model::channels_in"],
            out_dim=config["model::o_dim"],
            nlin=config["model::nlin"],
            dropout=config["model::dropout"],
            init_type=config["model::init_type"],
            width=config["model::width"],
        )
    elif config["model::type"] == "MiniAlexNet":
        logging.info("=> create MiniAlexNet")
        model = MiniAlexNet(
            channels_in=config["model::channels_in"],
            num_classes=config["model::o_dim"],
            init_type=config["model::init_type"],
        )
    elif config["model::type"] == "efficientnet_v2_s":
        logging.info("=> create efficientnet_v2_s")
        model = torchvision.models.efficientnet_v2_s(
            num_classes=config["model::o_dim"],
            dropout=config["model::dropout"],
        )
    elif config["model::type"] == "efficientnet_v2_m":
        logging.info("=> create efficientnet_v2_m")
        model = torchvision.models.efficientnet_v2_m(
            num_classes=config["model::o_dim"],
            dropout=config["model::dropout"],
        )
    elif config["model::type"] == "densenet121":
        logging.info("=> create densenet121")
        model = torchvision.models.densenet121(
            num_classes=config["model::o_dim"],
        )
    elif config["model::type"] == "densenet161":
        logging.info("=> create densenet161")
        model = torchvision.models.densenet161(
            num_classes=config["model::o_dim"],
        )
    elif config["model::type"] == "vit_b_16":
        logging.info("=> create vit_b_16")
        model = torchvision.models.vit_b_16(
            num_classes=config["model::o_dim"],
            dropout=config["model::dropout"],
        )
    elif config["model::type"] == "vit_l_16":
        logging.info("=> create vit_l_16")
        model = torchvision.models.vit_l_16(
            num_classes=config["model::o_dim"],
            dropout=config["model::dropout"],
        )
    elif config["model::type"] == "vit_s_16":
            logging.info("=> create vit_s_16")
            if config.get("training::mixup", None) is not None:
                mixup = config["training::mixup"]
                cutmix = config.get("training::cutmix", 0)
                rand_erase = config.get("training::random_erase", 0)
                mixup, rand_erase = get_data_augmentations(config.get("model::o_dim", 100), mixup, cutmix, rand_erase)
                self.use_mixup = True
            else:
                mixup = None
                rand_erase = None
                self.use_mixup = False

            model = ViTSmallPatch16(
                num_classes=config.get("model::o_dim", 100),
                init_type=config.get("model::init_type", "kaiming_normal"),
                fc_mlp=config.get("model::head::mlp", False),
                hidden_dim=config.get("model::head::hidden_dim", None),
                dropout=config.get("model::dropout", 0.),
                attn_dropout=config.get("model::attn_dropout", 0.),
                mixup=mixup,
                random_erase=rand_erase,
            )
    elif config["model::type"] == "ViT":
        logging.info("=> create ViT ")
        model = ViT.base(  # base config for VIT tiny
            num_classes=config["model::o_dim"],
            image_size=config.get("model::image_size", 32),
            patch_size=config.get("model::patch_size", 16),
            dim=config.get("model::embedding_dim", 192),
            depth=config.get("model::depth", 12),
            heads=config.get("model::heads", 3),
            mlp_dim=config.get("model::embedding_dim", 192) * 4,
            channels=config.get("model::channels_in", 3),
            dropout=config.get("model::dropout", 0.1),
        )
    else:
        raise NotImplementedError("error: model type unkown")

    return model


class IdentityTransform:
    def __call__(self, *args, **kwargs):
        # If only args were passed
        if args and not kwargs:
            return args if len(args) > 1 else args[0]
        # If only kwargs were passed
        elif kwargs and not args:
            return kwargs
        # If both args and kwargs were passed
        elif args and kwargs:
            return args if len(args) > 1 else args[0], kwargs
        # If nothing is passed (an edge case)
        return None

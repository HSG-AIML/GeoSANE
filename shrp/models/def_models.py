import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as ResNetBase

from shrp.models.def_resnet_width import ResNet_width

import logging


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
        # ASSUMES 28x28 image size
        # compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # add flatten layer
        self.module_list.append(nn.Flatten())
        # add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        # add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        # initialize weights with se methods
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
        # ASSUMES 28x28 image size
        # compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # compose layer 2
        self.module_list.append(nn.Conv2d(6, 9, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # compose layer 3
        self.module_list.append(nn.Conv2d(9, 6, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # add flatten layer
        self.module_list.append(nn.Flatten())
        # add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 6, 20))
        self.module_list.append(self.get_nonlin(nlin))
        # add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        # initialize weights with se methods
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
        # ASSUMES 32x32 image size
        # chn_in * 32 * 32
        # compose layer 0
        self.module_list.append(nn.Conv2d(channels_in, 16, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # 16 * 15 * 15
        # compose layer 1
        self.module_list.append(nn.Conv2d(16, 32, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # 32 * 7 * 7 // 32 * 6 * 6
        # compose layer 2
        self.module_list.append(nn.Conv2d(32, 15, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # 15 * 2 * 2
        self.module_list.append(nn.Flatten())
        # add linear layer 1
        self.module_list.append(nn.Linear(15 * 2 * 2, 20))
        self.module_list.append(self.get_nonlin(nlin))
        # add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        # initialize weights with se methods
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
        # ASSUMES 28x28 image size
        # input shape bx1x28x28

        # compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # output [15, 8, 12, 12]
        # residual connection stack 1
        self.res1_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res1_conv = nn.Conv2d(
            in_channels=channels_in, out_channels=8, kernel_size=1, stride=1, padding=0
        )

        # compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # output [15, 6, 4, 4]
        self.res2_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res2_conv = nn.Conv2d(
            in_channels=8, out_channels=6, kernel_size=1, stride=1, padding=0
        )

        # compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # output [15, 4, 3, 3]
        self.res3_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.res3_conv = nn.Conv2d(
            in_channels=6, out_channels=4, kernel_size=1, stride=1, padding=0
        )

        # add flatten layer
        self.module_list.append(nn.Flatten())
        # add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        # add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        # add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        # initialize weights with se methods
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

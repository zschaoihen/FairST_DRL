import math
import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.Tanh):
        super(ResidLinear, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

@gin.configurable('TemporalConvLayer')
class TemporalConvLayer(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, num_channels=gin.REQUIRED, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TemporalConvLayer, self).__init__()
        assert kernel_size % 2 == 1, "kernel size for the temproal gated network should be an odd number, otherwise cannot pad"
        num_pad = (kernel_size // 2)
        self.num_channels = num_channels
        self.conv1 = nn.Conv1d(num_channels, 2*num_channels, kernel_size, padding=num_pad)
        self.sigmoid = nn.Sigmoid()


    def forward(self, X):
        batch_size, n_frames_input, n_channels, H, W = X.size()
        X = X.permute(0, 3, 4, 2, 1)
        x_in = X.contiguous().view(-1, n_channels, n_frames_input)
        X = self.conv1(x_in)
        x_p = X[:, : self.num_channels, :]
        x_q = X[:, -self.num_channels:, :]

        x_glu = torch.mul((x_p + x_in), self.sigmoid(x_q))
        out = x_glu
        out = out.view(batch_size, H, W, n_channels, n_frames_input)
        # Convert back from NCHW to NHWC
        out = out.permute(0, 4, 3, 1, 2)
        return out

@gin.configurable('conv3x3')
def conv3x3(num_channels=gin.REQUIRED, kernel_size=gin.REQUIRED, norm=gin.REQUIRED):
    assert kernel_size % 2 == 1, "kernel size for the temproal gated network should be an odd number, otherwise cannot pad"
    num_pad = (kernel_size // 2)

    norm_layer = None
    if norm == 'BatchNorm2d':
        norm_layer = nn.BatchNorm2d
    elif norm == 'InstanceNorm2d':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'None':
        norm_layer = None
    else:
        raise NotImplementedError("Norm layer only support BN/IN/None")

    
    layers = [nn.Conv2d(num_channels, num_channels, kernel_size, 1, num_pad, bias=False)]
    if norm_layer:
        layers.append(norm_layer(num_channels))
    layers.append(nn.ReLU(True))

    layers.append(nn.Conv2d(num_channels, num_channels, kernel_size, 1, num_pad, bias=False))
    if norm_layer:
        layers.append(norm_layer(num_channels))

    return nn.Sequential(*layers)

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])


@gin.configurable('TemporalLSTMLayer')
class TemporalLSTMLayer(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, num_channels=gin.REQUIRED, num_layers=gin.REQUIRED):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param num_layers: num_layers of lstm layers.
        """
        super(TemporalLSTMLayer, self).__init__()
        self.num_channels = num_channels
        self.lstm = nn.LSTM(num_channels, num_channels, num_layers, batch_first=True)


    def forward(self, X):
        batch_size, n_frames_input, n_channels, H, W = X.size()
        assert n_channels * H * W == self.num_channels, "The features size does not fit!"
        X = X.view(batch_size, n_frames_input, self.num_channels)

        out, _ = self.lstm(X)
        out = out.view(batch_size, n_frames_input, n_channels, H, W)

        return out


@gin.configurable('TemporalAttentionLayer')
class TemporalAttentionLayer(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, num_channels=gin.REQUIRED, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TemporalAttentionLayer, self).__init__()
        assert kernel_size % 2 == 1, "kernel size for the temproal gated network should be an odd number, otherwise cannot pad"
        num_pad = (kernel_size // 2)
        self.num_channels = num_channels
        self.conv1 = nn.Conv1d(num_channels, 2*num_channels, kernel_size, padding=num_pad)
        self.sigmoid = nn.Sigmoid()


    def forward(self, X):
        batch_size, n_frames_input, n_channels, H, W = X.size()
        X = X.permute(0, 3, 4, 2, 1)
        x_in = X.contiguous().view(-1, n_channels, n_frames_input)
        X = self.conv1(x_in)
        x_p = X[:, : self.num_channels, :]
        x_q = X[:, -self.num_channels:, :]

        x_glu = torch.mul((x_p + x_in), self.sigmoid(x_q))
        out = x_glu
        out = out.view(batch_size, H, W, n_channels, n_frames_input)
        # Convert back from NCHW to NHWC
        out = out.permute(0, 4, 3, 1, 2)
        return out
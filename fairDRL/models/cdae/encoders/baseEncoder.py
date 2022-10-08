import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

@gin.configurable('BaseEncoder')
class BaseEncoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(BaseEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        activation = getattr(nn, activation)

        for i, (in_size, out_size) in enumerate(zip(layer_dim_list[:-2], layer_dim_list[1:-1])):
            self.layers.add_module(name="Linear{:d}".format(i), 
                                module=nn.Linear(in_size, out_size))
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))
        
        self.linear_means = nn.Linear(layer_dim_list[-2], layer_dim_list[-1])
        self.linear_log_var = nn.Linear(layer_dim_list[-2], layer_dim_list[-1])
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        means = self.linear_means(x)
        logvar = self.linear_log_var(x)
        return means, logvar


@gin.configurable('ConvEncoder')
class ConvEncoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(ConvEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the ConvEncoder, every element except the last one 
            should be a quadruple, and the last element should be a tuple 
            contains (z_dim and the except dimension after flatten).
        '''
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        activation = getattr(nn, activation)

        self.z_dim, self.view_size = layer_dim_list[-1]

        for i, dim_tuple in enumerate(layer_dim_list[:-1]):
            self.layers.add_module(name="Conv{:d}".format(i), 
                                    module=nn.Conv2d(*dim_tuple))
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))
        
        self.layers.add_module(name="Conv{:d}".format(num_layers), 
                                module=nn.Conv2d(self.view_size, 2*self.z_dim, 1))
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        means = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return means, logvar



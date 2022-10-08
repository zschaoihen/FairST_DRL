import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

@gin.configurable('BaseDecoder')
class BaseDecoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(BaseDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        activation = getattr(nn, activation)

        for i, (in_size, out_size) in enumerate(zip(layer_dim_list[:-2], layer_dim_list[1:-2])):
            self.layers.add_module(name="Linear{:d}".format(i), 
                                module=nn.Linear(in_size, out_size))
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))

        self.layers.add_module(name="Linear{:d}".format(num_layers), 
                                module=nn.Linear(layer_dim_list[-2], layer_dim_list[-1]))
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        return x

@gin.configurable('ConvDecoder')
class ConvDecoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(ConvDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the ConvEncoder, every element except the first one 
            should be a quadruple, and the first element should be a tuple 
            contains (z_dim and the except dimension after view).
        '''
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        activation = getattr(nn, activation)

        self.z_dim, self.view_size = layer_dim_list[0]

        self.layers.add_module(name="Conv{:d}".format(0), 
                                module=nn.Conv2d(self.z_dim, self.view_size, 1))
                                
        for i, dim_tuple in enumerate(layer_dim_list[1:]):
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))
            self.layers.add_module(name="ConvTranspose2d{:d}".format(i+1), 
                                    module=nn.ConvTranspose2d(*dim_tuple))
            
        # self.weight_init()

    
    def forward(self, x):
        x = self.layers(x)
        return x
import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

@gin.configurable('BaseGenerator')
class BaseGenerator(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(BaseGenerator, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        layers = []

        activation = getattr(nn, activation)

        for index in range(num_layers):
            layers.append(nn.Linear(layer_dim_list[index], layer_dim_list[index+1]))
            layers.append(activation(*act_param))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

@gin.configurable('BaseDiscriminator')
class BaseDiscriminator(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(BaseDiscriminator, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        layers = []

        activation = getattr(nn, activation)

        for index in range(num_layers):
            layers.append(nn.Linear(layer_dim_list[index], layer_dim_list[index+1]))
            layers.append(activation(*act_param))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

@gin.configurable('BaseRegressor')
class BaseRegressor(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(BaseRegressor, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        layers = []

        activation = getattr(nn, activation)

        for index in range(num_layers):
            layers.append(nn.Linear(layer_dim_list[index], layer_dim_list[index+1]))
            layers.append(activation(*act_param))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

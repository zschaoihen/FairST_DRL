import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .BaseDecoder import *

@gin.configurable('RevisedStDecoder')
class RevisedStDecoder(nn.Module):
    def __init__(self, spatial_channel=gin.REQUIRED, image_in_size=gin.REQUIRED,
                temporal_channel=gin.REQUIRED, image_decoder=gin.REQUIRED):
        super(RevisedStDecoder, self).__init__()

        self.image_decoder = image_decoder()
        self.adain = AdaIN()

        self.image_in_size = image_in_size
        self.spatial_channel = spatial_channel
        self.temporal_channel = temporal_channel
    
    def forward(self, spatial_z, temporal_z):
        batch_size, n_frames_input, _ = spatial_z.size()

        spatial_z = spatial_z.view(batch_size * n_frames_input, self.spatial_channel, self.image_in_size[0], self.image_in_size[1])
        temporal_z = temporal_z.view(batch_size * n_frames_input, self.temporal_channel, self.image_in_size[0], self.image_in_size[1])

        z = self.adain(spatial_z, temporal_z)
        x = self.image_decoder(z)
        _, num_channel, H, W = x.size()
        x = x.view(batch_size, n_frames_input, num_channel, H, W)
        return x

@gin.configurable('CDAE_3d_Decoder')
class CDAE_3d_Decoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(CDAE_3d_Decoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        
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

    def forward(self, x):
        x = self.layers(x)
        return x

@gin.configurable('CDAE_2d_Decoder')
class CDAE_2d_Decoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(CDAE_2d_Decoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        
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

    def forward(self, x):
        x = self.layers(x)
        return x

@gin.configurable('CDAE_1d_Decoder')
class CDAE_1d_Decoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(CDAE_1d_Decoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        
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

    def forward(self, x):
        x = self.layers(x)
        return x


@gin.configurable('FairDecoder')
class FairDecoder(nn.Module):
    def __init__(self, st_model=gin.REQUIRED, t_model=gin.REQUIRED, s_model=gin.REQUIRED, 
                        image_in_size=gin.REQUIRED, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(FairDecoder, self).__init__()

        self.st_model = st_model()
        self.t_model = t_model()
        self.s_model = s_model()

        self.image_in_size = image_in_size

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        
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

    def forward(self, z, sen):
        batch_size, n_frames_input, _ = z.size()
        z = z.view(batch_size, n_frames_input, 1,self.image_in_size[0], self.image_in_size[1])
        z = self.layers(z)

        z = torch.cat((z, sen), 2)

        x_3d = self.st_model(z)
        x_2d = self.t_model(z)
        x_1d = self.s_model(z)

        return (x_3d, x_2d, x_1d)
import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .baseEncoder import *

@gin.configurable('StEncoder')
class StEncoder(nn.Module):
    '''
    The backbone model. CNN + 2D LSTM.
    Given an input st_raster, output the mean and standard deviation of the latent vectors.
    '''
    def __init__(self, image_out_size=gin.REQUIRED, image_out_channel=gin.REQUIRED,  
                hidden_size=gin.REQUIRED, output_size=gin.REQUIRED, image_encoder=gin.REQUIRED):
        super(StEncoder, self).__init__()

        self.image_encoder = image_encoder()

        self.image_latent_size = image_out_size[0] * image_out_size[1] * image_out_channel
        # Encoder
        self.encode_rnn = nn.LSTM(self.image_latent_size, hidden_size,
                                    num_layers=1, batch_first=True)

        # Beta
        self.spatial_mu_layer = nn.Linear(self.image_latent_size, output_size)
        self.spatial_sigma_layer = nn.Linear(self.image_latent_size, output_size)

        self.temporal_mu_layer = nn.Linear(self.image_latent_size, output_size)
        self.temporal_sigma_layer = nn.Linear(self.image_latent_size, output_size)


        self.image_out_size = image_out_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, input):
        batch_size, n_frames_input, n_channels, H, W = input.size()
        # encode each frame
        input_reprs = self.image_encoder(input.view(-1, n_channels, H, W))
        input_reprs = input_reprs.view(batch_size, n_frames_input, -1)

        assert input_reprs.size() == (batch_size, n_frames_input, self.image_latent_size), "wrong latent size!"
        output, hidden = self.encode_rnn(input_reprs)

        input_reprs = input_reprs.view(-1, self.image_latent_size)
        output = output.contiguous().view(-1, self.hidden_size)

        spatial_mu = self.spatial_mu_layer(input_reprs).view(batch_size, n_frames_input, self.output_size)
        spatial_sigma = self.spatial_sigma_layer(input_reprs).view(batch_size, n_frames_input, self.output_size)
        temporal_mu = self.temporal_mu_layer(output).view(batch_size, n_frames_input, self.output_size)
        temporal_sigma = self.temporal_sigma_layer(output).view(batch_size, n_frames_input, self.output_size)

        return spatial_mu, spatial_sigma, temporal_mu, temporal_sigma

@gin.configurable('CDAE_3d_Encoder')
class CDAE_3d_Encoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(CDAE_3d_Encoder, self).__init__()

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

@gin.configurable('CDAE_2d_Encoder')
class CDAE_2d_Encoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(CDAE_2d_Encoder, self).__init__()

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

@gin.configurable('CDAE_1d_Encoder')
class CDAE_1d_Encoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(CDAE_1d, self).__init__()

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


@gin.configurable('FairEncoder')
class FairEncoder(nn.Module):
    def __init__(self, st_model=gin.REQUIRED, t_model=gin.REQUIRED, s_model=gin.REQUIRED, 
                        layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(FairEncoder, self).__init__()

        self.st_model = st_model()
        self.t_model = t_model()
        self.s_model = s_model()

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
        x_3d, x_2d, x_1d = x
        x_3d = self.st_model(x_3d)
        x_2d = self.t_model(x_2d)
        x_1d = self.s_model(x_1d)
        
        batch_size, n_frames_input, _, H, W = x_3d.size()
        # duplicate and concat
        # x_3d -> (batch, frame, channel, H, W)
        # x_2d -> (batch, channel, H, W)
        # x_1d -> (batch, channel, frame)

        x_2d = x_2d.repeat(1, n_frames_input, 1, 1, 1)
        x_1d = x_1d.repeat(1, 1, 1, H, W)
        x_1d = torch.transpose(x_1d, 1, 2)

        x = torch.cat((x_3d, x_2d, x_1d), 2)

        x = self.layers(x)

        means = x[:, :, :, :, 0].view(batch_size, -1)
        logvar = x[:, :, :, :, 1].view(batch_size, -1)

        sf = x[:, :, :, :, 2]
        return means, logvar, sf



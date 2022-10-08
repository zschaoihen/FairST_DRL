import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .encoders import *
from .decoders import *
from .others import *

@gin.configurable('CDAE_network')
class CDAE(nn.Module):
    def __init__(self, encoder=gin.REQUIRED, decoder=gin.REQUIRED):
                    
        super(CDAE, self).__init__()

        # assert type(encoder_layer_dims) == list
        # assert type(decoder_layer_dims) == list
        self.encoder = encoder()
        self.decoder = decoder()

    def reparameterize(self, mu, logvar):
        batch_size, n_frames_input, latent_size = mu.size()
        mu = mu.view(batch_size * n_frames_input, -1)
        logvar = logvar.view(batch_size * n_frames_input, -1)

        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())

        z = mu + std*eps
        return z.view(batch_size, n_frames_input, latent_size)

    def encode_(self, x):
        mu, sigma, sf = self.encoder(x)

        return mu, sigma, sf

    def decode_(self, z, sen):
        x = self.decoder(z, sen)
        return x

    def forward(self, x, sen, no_dec=False):
        mu, sigma, sf = self.encode_(x)
        
        z = self.reparameterize(mu, sigma)

        batch_size, n_frames_input, _ = z.size()
        
        if no_dec:
            z = torch.flatten(z, start_dim=1)
            return z

        else:
            x_recon = self.decode_(z, sen)
            z = torch.flatten(z, start_dim=1)
    
            mu = torch.flatten(mu, start_dim=1)
            sigma = torch.flatten(sigma, start_dim=1)   
            return x_recon, (mu, sigma), z
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from params import *

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( noise_size, filter_size*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(filter_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d( filter_size * 8,  filter_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size* 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(filter_size* 4, filter_size* 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size* 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d( filter_size*2,filter_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d( filter_size,n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(nc,discriminator_filter_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_filter_size, discriminator_filter_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_filter_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_filter_size* 2, discriminator_filter_size* 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_filter_size* 4, discriminator_filter_size* 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_filter_size* 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_filter_size* 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
  def __init__(self):
    super(Generator,self).__init__()
    self.c1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True))
    self.c2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=1, padding=1),
            nn.Upsample(scale_factor=2),

            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True))
    self.c3 = nn.Sequential(

            nn.ConvTranspose2d(64, 16, 4, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True))
    self.c4 = nn.Sequential(

            nn.ConvTranspose2d(16, 3, 4, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(3, 0.8),
            nn.LeakyReLU(0.2, inplace=True))
    self.th = nn.Tanh()
  def forward(self, z):
      out = self.c1(z)
      out = self.c2(out)
      out = self.c3(out)
      out = self.c4(out)
      img = self.th(out)
      return img


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=98):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 16 , 3)
        self.conv2 = nn.Conv2d(16,64,3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128,3)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.linear = nn.Linear(8192,1)
    def forward(self, input):
        x = F.leaky_relu(self.maxpool(self.conv1(input)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.maxpool(self.conv2(x))), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.maxpool(self.conv3(x))), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.maxpool(self.conv4(x))), 0.2)
        x = x.view(x.size(0),-1)
        x = F.sigmoid(self.linear(x))
        return x
gen = Generator().to(device)
dis = discriminator().to(device)

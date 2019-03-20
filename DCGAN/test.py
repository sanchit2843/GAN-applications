#download state dictionary
import wget
from params import *
from model import Generator
from utils import view_samples,im_convert
import torch
import argparse
from torch.autograd import Variable
import numpy as np
n = 20
n_col = 5
n_row = 4
wget.download(trained_weights_url)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#creating generator object
gen = Generator().to(device)
gen.load_state_dict(torch.load('./dcgan.h5' , map_location = device))

#define number of images to generate
Tensor = torch.FloatTensor
z = Variable(Tensor(np.random.normal(0, 1, (n,100,1,1))))
gen_imgs = gen(z)
a = []
for i in range(n):
  b = im_convert(gen_imgs[i])
  a.append(b)
#enter number of rows and columns in plot such that n_row*n_col=n
view_samples(n_row,n_col,a)

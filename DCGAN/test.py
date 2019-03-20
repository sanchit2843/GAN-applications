#download state dictionary
import wget
from params import *
from model import Generator
from utils import view_samples
wget.download(trained_weights_url)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#creating generator object
gen = Generator().to(device)
gen.load_state_dict(torch.load('./statedict.h5'))

#define number of images to generate
n = 20
z = Variable(Tensor(np.random.normal(0, 1, (n,100,1,1))))
gen_imgs = gen(z)
a = []
for i in range(n):
  b = im_convert(gen_imgs[i])
  a.append(b)
#enter number of rows and columns in plot such that n_row*n_col=n
view_samples(n_row,n_col,a)

from dataloader import dataloader
from model import Generator,Discriminator
import utils
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#generator object
gen = Generator().to(device)
#applying desired weights
gen.apply(utils.weights_init)
#discriminator object
dis = Discriminator().to(device)
#applying desired weights
dis.apply(utils.weights_init)
Tensor = torch.cuda.FloatTensor if 'cuda' else torch.FloatTensor
g_losses = []
d_losses = []

#define loss and optimizers
adversarial_loss = torch.nn.BCELoss()
optimizer_G = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(dis1.parameters(), lr=0.0002, betas=(0.5, 0.999))
#train
for epoch in range(1000):
    for i, imgs in enumerate(dataloader):

        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))
        optimizer_G.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0],100,1,1))))
        gen_imgs = gen(z)
        b = dis1(gen_imgs)
        g_loss = adversarial_loss(b, valid)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = adversarial_loss(dis1(real_imgs), valid)
        fake_loss = adversarial_loss(dis1(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    if(epoch%10==0):
      a = utils.im_convert(gen_imgs)
      plt.imshow(a)
      plt.show()
    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())
    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 1000, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))
error_plot(d_losses,g_losses)

import datacreation
from dataloader import dataloader
from model import Generator,Discriminator

Tensor = torch.cuda.FloatTensor if 'cuda' else torch.FloatTensor
for epoch in range(100):
    for i, imgs in enumerate(dataloader):

        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))
        optimizer_G.zero_grad()
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0],256,8,8))))
        # Generate a batch of images
        gen_imgs = gen(z)
        b = dis(gen_imgs)
        g_loss = adversarial_loss(b, valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = adversarial_loss(dis(real_imgs), valid)
        fake_loss = adversarial_loss(dis(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
    a = gen_imgs.cpu().detach().numpy()
    a = a[10]
    a = np.reshape(a,(158,158,3))
    imshow(a)
    show()
    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 100, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

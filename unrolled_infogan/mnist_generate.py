import argparse
import os

import torch
import torchvision.utils
import numpy as np
from models_mnist import Generator

parser = argparse.ArgumentParser()

parser.add_argument('--netG', type=str, default='',
    help="path to the weights of the generator network")
parser.add_argument('--netD', type=str, default='',
    help="path to the weights of the discriminator network")
parser.add_argument('--nZ', type=int, default=62, help='size of the noise')
parser.add_argument('--nDis', type=int, default=1, help='number of discrete latent variables')
parser.add_argument('--dDis', type=int, default=10, help='dimension of each discrete variable')
parser.add_argument('--nCon', type=int, default=2, help='number of continous latent variables')
parser.add_argument('--nC', type=int, default=1, help='number of image channels')
parser.add_argument('--nfG', type=int, default=64)
parser.add_argument('--nfD', type=int, default=64)
parser.add_argument('--disableCuda', action='store_true', help='disables cuda')
parser.add_argument('--outDir', type=str, default='output_unrolled_mnist_9',
    help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outDir)
except OSError:
    pass
try:
    os.makedirs(opt.outDir + '/samples')
except OSError:
    pass

useCuda = torch.cuda.is_available() and not opt.disableCuda
device = torch.device("cuda:0" if useCuda else "cpu")
Tensor = torch.cuda.FloatTensor if useCuda else torch.Tensor

# Networks.
G = Generator(nZ=opt.nZ, nDis=opt.nDis, dDis=opt.dDis, nCon=opt.nCon, nC=opt.nC, nfG=opt.nfG)\
    .to(device)

# Load the trained generator weights.
G.load_state_dict(torch.load("output_unrolled_mnist_9/checkpoints/G_epoch19.pth"))

c = np.linspace(-2, 2, 10).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)
c = torch.from_numpy(c).float().to(device)
c = c.view(-1, 1, 1, 1)

zeros = torch.zeros(100, 1, 1, 1, device=device)

# Continuous latent code.
c2 = torch.cat((c, zeros), dim=1)
c3 = torch.cat((zeros, c), dim=1)
# c2 = torch.cat((c, zeros, zeros, zeros), dim=1)
# c3 = torch.cat((zeros, c, zeros, zeros), dim=1)
# c4 = torch.cat((zeros, zeros, c, zeros), dim=1)
# c5 = torch.cat((zeros, zeros, zeros, c), dim=1)

idx = np.arange(10).repeat(10)
dis_c = torch.zeros(100, 10, 1, 1, device=device)
dis_c[torch.arange(0, 100), idx] = 1.0
# Discrete latent code.
c1 = dis_c.view(100, -1, 1, 1)

z = torch.randn(100, 62, 1, 1, device=device)

# To see variation along c2 (Horizontally) and c1 (Vertically)
noise1 = torch.cat((z, c1, c2), dim=1)
# To see variation along c3 (Horizontally) and c1 (Vertically)
noise2 = torch.cat((z, c1, c3), dim=1)
# noise3 = torch.cat((z, c1, c4), dim=1)
# noise4 = torch.cat((z, c1, c5), dim=1)

# Generate image.
with torch.no_grad():
    generated_img1 = G(noise1).detach().cpu()
# plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
torchvision.utils.save_image(generated_img1, opt.outDir + '/samples/sample_continous1.png',
    nrow=10, normalize=True)

# Generate image.
with torch.no_grad():
    generated_img2 = G(noise2).detach().cpu()
# plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1,2,0)))
torchvision.utils.save_image(generated_img2, opt.outDir + '/samples/sample_continous2.png',
    nrow=10, normalize=True)

# # Generate image.
# with torch.no_grad():
#     generated_img2 = G(noise3).detach().cpu()
# # plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1,2,0)))
# torchvision.utils.save_image(generated_img2, opt.outDir + '/samples/sample_continous3.png',
#     nrow=10, normalize=True)

# # Generate image.
# with torch.no_grad():
#     generated_img2 = G(noise4).detach().cpu()
# # plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1,2,0)))
# torchvision.utils.save_image(generated_img2, opt.outDir + '/samples/sample_continous4.png',
#     nrow=10, normalize=True)


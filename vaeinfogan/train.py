import argparse
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils

from dataset import CelebA
from models import *
from utils import weightsInit

parser = argparse.ArgumentParser()

parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
parser.add_argument('--lr', type=str, default='0.0001', help='learning rate')
parser.add_argument('--batchSize', type=int, default=64, help='batch size')
parser.add_argument('--latentSize', type=int, default=32, help='batch size')
parser.add_argument('--beta', type=int, default=100, help='the beta (hyper)parameter')
parser.add_argument('--nfE', type=int, default=64)
parser.add_argument('--nfD', type=int, default=512)
parser.add_argument('--resBlocks', type=str, default='0',
    help='number of residual blocks in both encoder and decoder')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of the optimizer')
parser.add_argument('--epochs', type=int, default=20, help='number of cycles over the data')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--disableCuda', action='store_true', help='disables cuda')
parser.add_argument('--outDir', type=str, default='output',
    help='folder to output images and model checkpoints')
parser.add_argument('--netE', type=str, default='',
    help="path to the encoder network (to continue training)")
parser.add_argument('--netD', type=str, default='',
    help="path to the decoder network (to continue training)")

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outDir)
except OSError:
    pass
try:
    os.makedirs(opt.outDir + '/running_samples')
except OSError:
    pass
try:
    os.makedirs(opt.outDir + '/checkpoints')
except OSError:
    pass

useCuda = torch.cuda.is_available() and not opt.disableCuda
device = torch.device("cuda:0" if useCuda else "cpu")
Tensor = torch.cuda.FloatTensor if useCuda else torch.Tensor

# Networks.
E = Encoder(latentSize=opt.latentSize, nfE=opt.nfE, resBlocks=int(opt.resBlocks)).to(device)
R = Reparametrization(latentSize=opt.latentSize).to(device)
D = Decoder(latentSize=opt.latentSize, nfD=opt.nfD, resBlocks=int(opt.resBlocks)).to(device)

# Initialize the weights.
E.apply(weightsInit)
D.apply(weightsInit)

# Define the Reconstruction loss.
# We will add KL divergence while training.
reconstructionLoss = nn.MSELoss(reduction='sum')

# Optimizers.
optimizer = getattr(torch.optim, opt.optimizer)
optimizerE = optimizer(E.parameters(), lr=float(opt.lr))
optimizerD = optimizer(D.parameters(), lr=float(opt.lr))

# Path to data.
imgDirectory = '../../Downloads/img_align_celeba/'

# Create dataloader.
dataloader = DataLoader(CelebA(imgDirectory), batch_size=opt.batchSize, shuffle=True,
    num_workers=8)

# Lists to keep track of progress.
klLossList = []
reconLossList = []
samplesPriorList = []

print('Training started.')
for epoch in range(opt.epochs):
    for i, batch in enumerate(dataloader):
        E.zero_grad()
        D.zero_grad()

        # Forward pass.
        output = E(batch.type(Tensor))
        output = R(output)
        output = D(output)

        # Evaluate losses.(KL divergence and Reconstruction loss)
        klLoss = opt.beta * R.klDivergence()
        reconLoss = reconstructionLoss(output, batch.type(Tensor)) / batch.shape[0]

        reconLoss.backward(retain_graph=True)
        klLoss.backward()

        optimizerE.step()
        optimizerD.step()

        if i % 50 == 0:
            print('Epoch -> ' + str(epoch) + ', Batch -> ' + str(i))
            print('Reconstruction loss -> ' + str(reconLoss.item()))
            print('KL loss -> ' + str(klLoss.item()))

        # Save the losses.
        reconLossList.append(reconLoss.item())
        klLossList.append(klLoss.item())

        if i % 500 == 0:
            samples = D(torch.randn(64, opt.latentSize).type(Tensor)).detach()

            samplesPriorList.append(torchvision.utils.make_grid(samples, padding=2,
                normalize=True))
            torchvision.utils.save_image(samples, opt.outDir + '/running_samples/' + 'sample_' +
                str(epoch) + '_' + str(i) + '.png', normalize=True)

            # Save the lists.
            pickle_out = open(opt.outDir + '/lists.pickle', 'wb')
            pickle.dump([klLossList, reconLossList, samplesPriorList, opt], pickle_out)
            pickle_out.close()

    # Checkpoints.
    torch.save(E.state_dict(), opt.outDir + '/checkpoints/E_epoch' + str(epoch) + '.pth')
    torch.save(D.state_dict(), opt.outDir + '/checkpoints/D_epoch' + str(epoch) + '.pth')

# Save the lists.
pickle_out = open(opt.outDir + '/lists.pickle', 'wb')
pickle.dump([klLossList, reconLossList, samplesPriorList, opt], pickle_out)
pickle_out.close()

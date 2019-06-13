import argparse
import os
import pickle
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils

from dataset import CelebA
from models import *
from utils import weightsInit

parser = argparse.ArgumentParser()

parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
parser.add_argument('--lr', type=str, default='0.00005', help='learning rate')
parser.add_argument('--batchSize', type=int, default=64, help='batch size')
parser.add_argument('--latentSize', type=int, default=256, help='batch size')
parser.add_argument('--unrollSteps', type=int, default=5, help='unrolling steps')
parser.add_argument('--beta', type=int, default=50, help='the beta (hyper)parameter')
parser.add_argument('--nfE', type=int, default=64)
parser.add_argument('--nfD', type=int, default=512)
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of the optimizer')
parser.add_argument('--epochs', type=int, default=20, help='number of cycles over the data')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--disableCuda', action='store_true', help='disables cuda')
parser.add_argument('--outDir', type=str, default='output_unrolled',
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
E = Encoder(latentSize=opt.latentSize, nfE=opt.nfE).to(device)
R = Reparametrization(latentSize=opt.latentSize).to(device)
D = Decoder(latentSize=opt.latentSize, nfD=opt.nfD).to(device)
Dis = Discriminator().to(device)

# Initialize the weights.
E.apply(weightsInit)
D.apply(weightsInit)
Dis.apply(weightsInit)

# Load weights if provided.
# if opt.netE != '':
# E.load_state_dict(torch.load("output_unrolled/checkpoints/E_epoch2.pth"))
# # if opt.netD != '':
# D.load_state_dict(torch.load("output_unrolled/checkpoints/D_epoch2.pth"))
# # if opt.netDis != '':
# Dis.load_state_dict(torch.load("output_unrolled/checkpoints/Dis_epoch2.pth"))

# Define the Reconstruction loss.
# We will add KL divergence while training.
reconstructionLoss = nn.MSELoss(reduction='sum')
binaryCrossEntropy = nn.BCELoss()

# Optimizers.
optimizer = getattr(torch.optim, opt.optimizer)
optimizerE = optimizer(E.parameters(), lr=float(0.0005))
optimizerD = optimizer(D.parameters(), lr=float(0.0005))
optimizerDis = optimizer(Dis.parameters(), lr=float(0.00005), betas=(0.9, 0.999))

# Path to data.
imgDirectory = '../../Downloads/img_align_celeba/img_align_celeba/'

# Create dataloader.
dataloader = DataLoader(CelebA(imgDirectory), batch_size=opt.batchSize, shuffle=True,
    num_workers=opt.workers)

# Lists to keep track of progress.
klLossList = []
# reconLossList = []
samplesPriorList = []
realLabel = 1
fakeLabel = 0

dataloaderIter = iter(dataloader)

print('Training started.')
for epoch in range(opt.epochs):
    for i, batch in enumerate(dataloader):
        batchShape = batch.shape[0]

        Dis.zero_grad()

        outputRealDis = Dis(batch.type(Tensor))
        label = torch.full((batchShape,), realLabel).type(Tensor)
        lossRealDis = binaryCrossEntropy(outputRealDis, label)
        lossRealDis.backward()

        outputED = E(batch.type(Tensor))
        outputED = R(outputED)
        outputED = D(outputED)

        outputFakeDis = Dis(outputED.detach())
        label = label.fill_(fakeLabel)
        lossFakeDis = binaryCrossEntropy(outputFakeDis, label)
        lossFakeDis.backward()

        lossDis = lossRealDis + lossFakeDis

        optimizerDis.step()

        if opt.unrollSteps > 0:
            CopyDis = copy.deepcopy(Dis)

            for l in range(opt.unrollSteps):
                Dis.zero_grad() # Check.

                try:
                    data = next(dataloaderIter)
                except StopIteration:
                    dataloaderIter = iter(dataloader)
                    data = next(dataloaderIter)

                dataShape = data.shape[0]
                if dataShape != batchShape:
                    continue

                outputRealDis = Dis(batch.type(Tensor))
                label = torch.full((batchShape,), realLabel).type(Tensor)
                lossRealDis = binaryCrossEntropy(outputRealDis, label)
                lossRealDis.backward(create_graph=True)

                with torch.no_grad():
                    outputED = E(batch.type(Tensor))
                    outputED = R(outputED)
                    outputED = D(outputED)

                # outputFakeDis = Dis(outputE.detach())
                outputFakeDis = Dis(outputED)
                label = label.fill_(fakeLabel)
                lossFakeDis = binaryCrossEntropy(outputFakeDis, label)
                lossFakeDis.backward(create_graph=True)

                # lossDis = lossRealDis + lossFakeDis
                optimizerDis.step()

        E.zero_grad()
        D.zero_grad()

        # Forward pass.
        output = E(batch.type(Tensor))
        output = R(output)
        output = D(output)

        outputDis = Dis(output)
        label = label.fill_(realLabel)
        lossG = binaryCrossEntropy(outputDis, label)
        lossG.backward(retain_graph=True)


        # Evaluate losses.(KL divergence and Reconstruction loss)
        klLoss = opt.beta * R.klDivergence()
        # reconLoss = 0.0025 * reconstructionLoss(output, batch.type(Tensor)) / batch.shape[0]

        # reconLoss.backward(retain_graph=True)
        klLoss.backward()

        optimizerE.step()
        optimizerD.step()

        if opt.unrollSteps > 0:
            Dis.load(CopyDis)
            del CopyDis


        if i % 20 == 0:
            print('Epoch -> ' + str(epoch) + ', Batch -> ' + str(i))
            # print('Reconstruction loss -> ' + str(reconLoss.item()))
            print('KL loss -> ' + str(klLoss.item()))
            print('Discriminator loss -> ' + str(lossDis.item()))
            print('Generator loss -> ' + str(lossG.item()))

        # Save the losses.
        # reconLossList.append(reconLoss.item())
        klLossList.append(klLoss.item())

        if i % 200 == 0:
            samples = D(torch.randn(64, opt.latentSize).type(Tensor)).detach()

            samplesPriorList.append(torchvision.utils.make_grid(samples, padding=2,
                normalize=True))
            torchvision.utils.save_image(samples, opt.outDir + '/running_samples/' + 'sample_' +
                str(epoch) + '_' + str(i) + '.png', normalize=True)

            # Save the lists.
            pickle_out = open(opt.outDir + '/lists.pickle', 'wb')
            # pickle.dump([klLossList, reconLossList, samplesPriorList, opt], pickle_out)
            pickle_out.close()

    # Checkpoints.
    torch.save(E.state_dict(), opt.outDir + '/checkpoints/E_epoch' + str(epoch) + '.pth')
    torch.save(D.state_dict(), opt.outDir + '/checkpoints/D_epoch' + str(epoch) + '.pth')
    torch.save(Dis.state_dict(), opt.outDir + '/checkpoints/Dis_epoch' + str(epoch) + '.pth')

# Save the lists.
pickle_out = open(opt.outDir + '/lists.pickle', 'wb')
# pickle.dump([klLossList, reconLossList, samplesPriorList, opt], pickle_out)
pickle_out.close()

import argparse
import os
import pickle
import time
import copy
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils

from models import Generator, Discriminator
from utils import weightsInit, NormalNLLLoss, noiseSample
from dataset import Nuclei

parser = argparse.ArgumentParser()

parser.add_argument('--lrG', type=float, default=1e-3, help='learning rate of generatorEG')
parser.add_argument('--lrQ', type=float, default=1e-3, help='learning rate of Q')
parser.add_argument('--lrD', type=float, default=2e-4, help='learning rate of discriminatorD')
parser.add_argument('--unrollSteps', type=int, default=5, help='unrolling steps')
parser.add_argument('--batchSize', type=int, default=64, help='batch size')
parser.add_argument('--nZ', type=int, default=32, help='size of the noise')
parser.add_argument('--nDis', type=int, default=1, help='number of discrete latent variables')
parser.add_argument('--dDis', type=int, default=5, help='dimension of each discrete variable')
parser.add_argument('--nCon', type=int, default=3, help='number of continous latent variables')
parser.add_argument('--nC', type=int, default=3, help='number of image channels')
parser.add_argument('--nfG', type=int, default=64)
parser.add_argument('--nfD', type=int, default=64)
parser.add_argument('--epochs', type=int, default=40,
    help='number of complete cycles over the data')
parser.add_argument('--seed', type=int, default=79, help='random seed')
parser.add_argument('--workers', type=int, default=6, help='number of data loading CPU cores')
parser.add_argument('--disableCuda', action='store_true', help='disables cuda')
parser.add_argument('--outDir', type=str, default='output_unrolled_nuclei_2',
    help='folder to output samples, losses and model checkpoints')
parser.add_argument('--netG', type=str, default='',
    help="path to the generator network (to continue training)")
parser.add_argument('--netD', type=str, default='',
    help="path to the discriminator network (to continue training)")

opt = parser.parse_args()
print(opt)

# Random seed for reproducibility.
random.seed(opt.seed)
torch.manual_seed(opt.seed)

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
G = Generator(nZ=opt.nZ, nDis=opt.nDis, dDis=opt.dDis, nCon=opt.nCon, nC=opt.nC, nfG=opt.nfG)\
    .to(device)
D = Discriminator(nC=opt.nC, nfD=opt.nfD, nZ=opt.nZ, nDis=opt.nDis, dDis=opt.dDis, nCon=opt.nCon)\
    .to(device)

# Initialize the weights.
G.apply(weightsInit)
D.apply(weightsInit)

# Load weights if provided.
if opt.netG != '':
    G.load_state_dict(torch.load(opt.netG))
if opt.netD != '':
    D.load_state_dict(torch.load(opt.netD))

# Define the losses.
binaryCrossEntropy = nn.BCELoss() # Discriminate between real and fake images.
crossEntropy = nn.CrossEntropyLoss() # Loss for discrete latent variables.
normalNLL = NormalNLLLoss() # Loss for continuous latent varibales.

# Optimizers.
optimizerG = torch.optim.Adam(G.parameters(), lr=opt.lrG, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(list(D.commonDQ.parameters()) + list(D.finalD.parameters()),
    lr=opt.lrD, betas=(0.5, 0.999))
optimizerQ = torch.optim.Adam(list(D.finalQ.parameters()) + list(G.parameters()), lr=opt.lrQ, betas=(0.5, 0.999))
# optimizerQ = torch.optim.Adam(D.finalQ.parameters(), lr=opt.lrQ, betas=(0.5, 0.999))

# Path to data.
imgDirectory = '../../tcga_unnormalized_images/nuclei_aligned/'
# imgDirectory = "../../Downloads/img_align_celeba/"
# imgDirectory = "../../Downloads/mnist/fullMnist/"

# Create dataloader.
dataloader = DataLoader(Nuclei(imgDirectory), batch_size=opt.batchSize, shuffle=True,
    num_workers=opt.workers)

realLabel = 1
fakeLabel = 0

noiseFixed, _ = noiseSample(opt.nZ, opt.nDis, opt.dDis, opt.nCon, opt.batchSize)
noiseFixed = noiseFixed.type(Tensor)

# Lists to keep track of progress.
lossesD = []
lossesG = []
lossesQDis = []
lossesQCon = []

dataloaderIter = iter(dataloader)

startTime = time.time()
for epoch in range(opt.epochs):
    for i, batch in enumerate(dataloader):
        batchShape = batch.shape[0]

        # Train the discriminator:
        D.zero_grad()

        outputRealD = D.discriminate(batch.type(Tensor))

        label = torch.full((batchShape,), realLabel).type(Tensor)
        lossRealD = binaryCrossEntropy(outputRealD, label)

        lossRealD.backward(retain_graph=True)

        noise, _ = noiseSample(opt.nZ, opt.nDis, opt.dDis, opt.nCon, batchShape)
        noise = noise.type(Tensor)

        outputG = G(noise)
        outputFakeD = D.discriminate(outputG.detach())

        label = label.fill_(fakeLabel)
        lossFakeD = binaryCrossEntropy(outputFakeD, label)

        lossFakeD.backward()

        lossD = lossRealD + lossFakeD

        optimizerD.step()

        # Train the generator:

        noise, target = noiseSample(opt.nZ, opt.nDis, opt.dDis, opt.nCon, batchShape)
        noise = noise.type(Tensor)
        target = target.type(torch.cuda.LongTensor)

        if opt.unrollSteps > 0:
            copyD = copy.deepcopy(D)

            for j in range(opt.unrollSteps):
                D.zero_grad()

                try:
                    batchUnroll = next(dataloaderIter)
                except StopIteration:
                    dataloaderIter = iter(dataloader)
                    batchUnroll = next(dataloaderIter)

                if batchUnroll.shape[0] != batchShape:
                    continue

                outputRealD = D.discriminate(batchUnroll.type(Tensor))
                label = torch.full((batchShape,), realLabel).type(Tensor)
                lossRealD = binaryCrossEntropy(outputRealD, label)
                lossRealD.backward(create_graph=True)
                # noise, _ = noiseSample(opt.nZ, opt.nDis, opt.dDis, opt.nCon, batchShape)
                noise = noise.type(Tensor)
                with torch.no_grad():
                    outputG = G(noise)
                # outputG = G(noise)

                outputFakeD = D.discriminate(outputG) # Try detach here without no_grad() above.
                label = label.fill_(fakeLabel)
                lossFakeD = binaryCrossEntropy(outputFakeD, label)
                lossFakeD.backward(create_graph=True)

                optimizerD.step()

        G.zero_grad()
        D.zero_grad()

        # noise, target = noiseSample(opt.nZ, opt.nDis, opt.dDis, opt.nCon, batchShape)
        # noise = noise.type(Tensor)
        # target = target.type(torch.cuda.LongTensor)

        outputG = G(noise)
        # outputD, disLogits, conMean, conStd = D(outputG)
        outputD = D.discriminate(outputG)

        label = label.fill_(realLabel)
        lossG = binaryCrossEntropy(outputD, label)

        # lossQDis = 0
        # for j in range(opt.nDis):
        #     lossQDis += crossEntropy(disLogits[:, j * opt.dDis: j * opt.dDis + opt.dDis],
        #         target[j])

        # lossQCon = 0
        # if (opt.nCon != 0):
        #     lossQCon = normalNLL(noise[:, opt.nZ + opt.nDis * opt.dDis: ].view(-1, opt.nCon),\
        #         conMean, conStd) * 0.1

        # lossGQ = lossG + lossQDis + lossQCon
        # lossGQ.backward()
        lossG.backward()

        optimizerG.step()

        if opt.unrollSteps > 0:
            D.load(copyD)
            del copyD

        G.zero_grad()
        D.zero_grad()

        outputG = G(noise)
        disLogits, conMean, conStd = D.outputQ(outputG)

        lossQDis = 0
        for j in range(opt.nDis):
            lossQDis += crossEntropy(disLogits[:, j * opt.dDis: j * opt.dDis + opt.dDis],
                target[j])

        lossQCon = 0
        if (opt.nCon != 0):
            lossQCon = normalNLL(noise[:, opt.nZ + opt.nDis * opt.dDis: ].view(-1, opt.nCon),\
                conMean, conStd) * 0.1
        lossQ = lossQDis + lossQCon
        lossQ.backward()

        optimizerQ.step()
        # optimizerG.step()

        if i % 50 == 0:
            print('Epoch -> ' + str(epoch) + ', Batch -> ' + str(i))
            print('Discriminator loss -> ' + str(lossD.item()))
            print('Generator loss -> ' + str(lossG.item()))
            print('Q discrete loss -> ' + str(lossQDis.item()))
            print('Q continuous loss -> ' + str(lossQCon.item()))

        # Save the losses.
        lossesD.append(lossD.item())
        lossesG.append(lossG.item())
        lossesQDis.append(lossQDis.item())
        lossesQCon.append(lossQCon.item())

        if i % 500 == 0:
            generatedSample = G(noiseFixed).detach()

            torchvision.utils.save_image(generatedSample, opt.outDir + '/running_samples/'
                + 'sample_' + str(epoch) + '_' + str(i) + '.png', normalize=True)

            # Save the lists(keep overwriting, we only need the latest).
            pickle_out = open(opt.outDir + '/lists.pickle', 'wb')
            pickle.dump([lossesD, lossesG, lossesQDis, lossesQCon, opt], pickle_out)
            pickle_out.close()

    # Checkpoints.
    torch.save(G.state_dict(), opt.outDir + '/checkpoints/G_epoch' + str(epoch) + '.pth')
    torch.save(D.state_dict(), opt.outDir + '/checkpoints/D_epoch' + str(epoch) + '.pth')

    print('Time elapsed: ' + str(time.time() - startTime) + 'seconds')

# Save the lists(keep overwriting, we only need the latest).
pickle_out = open(opt.outDir + '/lists.pickle', 'wb')
pickle.dump([lossesD, lossesG, lossesQDis, lossesQCon, opt], pickle_out)
pickle_out.close()



import argparse
import os
import pickle
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils

from dataset import CelebA
from models_vaegan import *
from utils import weightsInit

parser = argparse.ArgumentParser()

parser.add_argument('--lrE', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lrD', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lrDis', type=float, default=2e-4, help='learning rate')
parser.add_argument("--decayLr", type=float, default=1, help="learning rate decay rate")
parser.add_argument('--batchSize', type=int, default=64, help='batch size')
parser.add_argument('--latentSize', type=int, default=64, help='batch size')
parser.add_argument('--unrollSteps', type=int, default=5, help='unrolling steps')
parser.add_argument('--beta', type=int, default=100, help='the beta (hyper)parameter')
parser.add_argument("--lambd", type=float, default=1e-3, help="lambda to weight feature-wise loss")
parser.add_argument('--nfE', type=int, default=64)
parser.add_argument('--nfD', type=int, default=64)
parser.add_argument('--nfDis', type=int, default=64)
parser.add_argument('--epochs', type=int, default=20, help='number of cycles over the data')
parser.add_argument('--workers', type=int, default=6, help='number of data loading workers')
parser.add_argument('--disableCuda', action='store_true', help='disables cuda')
parser.add_argument('--outDir', type=str, default='output_unrolled_beta250_5',
    help='folder to output images and model checkpoints')
parser.add_argument('--netE', type=str, default='',
    help="path to the encoder network (to continue training)")
parser.add_argument('--netD', type=str, default='',
    help="path to the decoder network (to continue training)")
parser.add_argument('--netDis', type=str, default='',
    help="path to the decoder network (to continue training)")

opt = parser.parse_args()
print(opt)

torch.manual_seed(8)
torch.cuda.manual_seed(8)

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
Dis = Discriminator(nfDis=opt.nfDis).to(device)

# Initialize the weights.
E.apply(weightsInit)
D.apply(weightsInit)
Dis.apply(weightsInit)

# Load weights if provided.
if opt.netE != '':
    E.load_state_dict(torch.load(opt.netE))
if opt.netD != '':
    D.load_state_dict(torch.load(opt.netD))
if opt.netDis != '':
    Dis.load_state_dict(torch.load(opt.netDis))

# Define the losses.
# We will add KL divergence while training.
binaryCrossEntropy = nn.BCELoss(reduction='sum') # GAN loss.
meanSquaredError = nn.MSELoss(reduction='sum') # Feature wise loss.

# Define the optimizers.
# optimizerE = torch.optim.Adam(E.parameters(), lr=opt.lrE, betas=(0.5, 0.999))
# optimizerD = torch.optim.Adam(D.parameters(), lr=opt.lrD, betas=(0.5, 0.999))
# optimizerDis = torch.optim.Adam(Dis.parameters(), lr=opt.lrDis, betas=(0.5, 0.999))
optimizerE = torch.optim.RMSprop(E.parameters(), lr=opt.lrE, alpha=0.9, eps=1e-8)
optimizerD = torch.optim.RMSprop(D.parameters(), lr=opt.lrD, alpha=0.9, eps=1e-8)
optimizerDis = torch.optim.RMSprop(Dis.parameters(), lr=opt.lrDis, alpha=0.9, eps=1e-8)

lrSchedulerE = torch.optim.lr_scheduler.ExponentialLR(optimizerE, opt.decayLr)
lrSchedulerD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, opt.decayLr)
lrSchedulerDis = torch.optim.lr_scheduler.ExponentialLR(optimizerDis, opt.decayLr)

# Path to data.
imgDirectory = "../../Downloads/img_align_celeba/"

# Create dataloader.
dataloader = DataLoader(CelebA(imgDirectory), batch_size=opt.batchSize, shuffle=True,
    num_workers=opt.workers)

testBatch = next(iter(dataloader)).type(Tensor)
torchvision.utils.save_image(testBatch, opt.outDir + '/testBatch.png', normalize=True)
testLatent = torch.randn(64, opt.latentSize).type(Tensor)

# Lists to keep track of progress.
lossesKl = [] # KL divergence of the encoded distribution from the prior.
lossesD = [] # Decoder/Generator losses from the GAN objective.
lossesDis = [] # Discriminator losses from the GAN objective.
lossesDisL = [] # Feature-wise reconstruction losses of the encoder-decoder pair.

realLabel = 1
fakeLabel = 0

dataloaderIter = iter(dataloader)

print('Training started.')
for epoch in range(opt.epochs):
    for i, batch in enumerate(dataloader):
        batchShape = batch.shape[0]
        samplePrior = torch.randn(batchShape, opt.latentSize).type(Tensor)

        Dis.zero_grad()

        outputRealDis = Dis(batch.type(Tensor))

        label = torch.full((batchShape,), realLabel).type(Tensor)
        # lossRealDis = binaryCrossEntropy(outputRealDis, label) / batchShape
        lossRealDis = binaryCrossEntropy(outputRealDis, label)
        lossRealDis.backward() # Why is retain_graph not needed here?

        outputReconD = D(R(E(batch.type(Tensor)))) # Should a random prior sample be used here? -> answer: both
        outputReconDis = Dis(outputReconD.detach())

        label = label.fill_(fakeLabel)
        lossReconDis = binaryCrossEntropy(outputReconDis, label)

        outputPriorD = D(samplePrior)
        outputPriorDis = Dis(outputPriorD.detach())

        lossPriorDis = binaryCrossEntropy(outputPriorDis, label)

        # lossFakeDis = (lossPriorDis + lossReconDis) / batchShape
        lossFakeDis = lossPriorDis + lossReconDis
        lossFakeDis.backward()

        lossDis = lossRealDis + lossFakeDis

        optimizerDis.step()

        Dis.clearFeatures()
        # samplePrior = torch.randn(batchShape, opt.latentSize).type(Tensor)
        if opt.unrollSteps > 0:
            CopyDis = copy.deepcopy(Dis)

            for l in range(opt.unrollSteps):
                Dis.zero_grad() # Check.

                try:
                    batchUnroll = next(dataloaderIter)
                except StopIteration:
                    dataloaderIter = iter(dataloader)
                    batchUnroll = next(dataloaderIter)

                if batchUnroll.shape[0] != batchShape:
                    continue

                outputRealDis = Dis(batchUnroll.type(Tensor))
                label = torch.full((batchShape,), realLabel).type(Tensor)
                # lossRealDis = binaryCrossEntropy(outputRealDis, label) / batchShape
                lossRealDis = binaryCrossEntropy(outputRealDis, label)
                lossRealDis.backward(create_graph=True)

                samplePrior = torch.randn(batchShape, opt.latentSize).type(Tensor)
                with torch.no_grad():
                    outputReconD = D(R(E(batchUnroll.type(Tensor))))
                    outputPriorD = D(samplePrior)

                outputReconDis = Dis(outputReconD)
                outputPriorDis = Dis(outputPriorD)

                label = label.fill_(fakeLabel)
                lossReconDis = binaryCrossEntropy(outputReconDis, label)
                lossPriorDis = binaryCrossEntropy(outputPriorDis, label)

                lossFakeDis = lossPriorDis + lossReconDis
                lossFakeDis.backward(create_graph=True)

                optimizerDis.step()

        D.zero_grad()

        outputReconD = D(R(E(batch.type(Tensor)))) # Check if you can reduce any repeated forward passes.
        outputReconDis = Dis(outputReconD)
        # featuresRecon = Dis.getFeatures()

        samplePrior = torch.randn(batchShape, opt.latentSize).type(Tensor)
        outputPriorD = D(samplePrior)
        outputPriorDis = Dis(outputPriorD)
        # featuresPrior = Dis.getFeatures()

        label = label.fill_(realLabel)
        # lossGanD = (binaryCrossEntropy(outputPriorDis, label) + \
        #     binaryCrossEntropy(outputReconDis, label)) / batchShape
        lossGanD = binaryCrossEntropy(outputPriorDis, label) + \
            binaryCrossEntropy(outputReconDis, label)

        # featuresReal = Dis.getFeatures(batch.type(Tensor), savedFeatures=False)
        # lossDisLRecon = meanSquaredError(featuresRecon, featuresReal.detach())
        # lossDisLPrior = meanSquaredError(featuresPrior, featuresReal.detach())

        # lossD = lossGanD + (lossDisLRecon + lossDisLPrior) * opt.lambd
        # lossD.backward(retain_graph=True)
        lossGanD.backward(retain_graph=True)

        # optimizerD.step()

        if opt.unrollSteps > 0:
            Dis.load(CopyDis)
            del CopyDis

        featuresRecon = Dis.getFeatures(outputReconD, False)
        featuresPrior = Dis.getFeatures(outputPriorD, False)
        featuresReal = Dis.getFeatures(batch.type(Tensor), False)

        lossDisLRecon = meanSquaredError(featuresRecon, featuresReal.detach())
        lossDisLPrior = meanSquaredError(featuresPrior, featuresReal.detach())

        lossDDisL = (lossDisLRecon + lossDisLPrior) * opt.lambd
        lossDDisL.backward(retain_graph=True)

        optimizerD.step()

        E.zero_grad()

        lossKl = opt.beta * R.klDivergence(mean=False)
        lossE = lossKl + lossDisLRecon
        lossE.backward()

        optimizerE.step()


        if i % 20 == 0:
            print('Epoch -> ' + str(epoch) + ', Batch -> ' + str(i))
            print('DisL loss -> ' + str(lossDisLRecon.item() / batchShape) + ' ' + \
                str(lossDisLPrior.item() / batchShape))
            print('KL loss -> ' + str(lossKl.item() / batchShape))
            print('Discriminator loss -> ' + str(lossDis.item() / batchShape))
            print('Decoder(Generator) loss -> ' + str(lossGanD.item() / batchShape))

        # Save the losses.
        lossesDisL.append((lossDisLRecon / batchShape).item())
        lossesKl.append((lossKl / batchShape).item())
        lossesD.append((lossGanD / batchShape).item())
        lossesDis.append((lossDis / batchShape).item())

        if i % 200 == 0:
            samplesPrior = D(testLatent).detach()
            samplesPosterior = D(R(E(testBatch)))

            torchvision.utils.save_image(samplesPrior, opt.outDir + '/running_samples/' + 'sample_prior' +
                str(epoch) + '_' + str(i) + '.png', normalize=True)
            torchvision.utils.save_image(samplesPosterior, opt.outDir + '/running_samples/' + 'sample_posterior' +
                str(epoch) + '_' + str(i) + '.png', normalize=True)

            # Save the lists.
            pickle_out = open(opt.outDir + '/lists.pickle', 'wb')
            pickle.dump([lossesDisL, lossesKl, lossesD, lossesDis, opt], pickle_out)
            pickle_out.close()

    # Checkpoints.
    torch.save(E.state_dict(), opt.outDir + '/checkpoints/E_epoch' + str(epoch) + '.pth')
    torch.save(D.state_dict(), opt.outDir + '/checkpoints/D_epoch' + str(epoch) + '.pth')
    torch.save(Dis.state_dict(), opt.outDir + '/checkpoints/Dis_epoch' + str(epoch) + '.pth')

    lrSchedulerE.step()
    lrSchedulerD.step()
    lrSchedulerDis.step()

# Save the lists.
pickle_out = open(opt.outDir + '/lists.pickle', 'wb')
pickle.dump([lossesDisL, lossesKl, lossesD, lossesDis, opt], pickle_out)
pickle_out.close()


import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
# Spectral norm doesn't work with create_graph=True, throws error, check why.

# Create a layer to reshape within Sequential layers, for convenience.
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Generator(nn.Module):
    def __init__(self, nZ=128, nDis=10, dDis=10, nCon=0, nC=3, nfG=64):
        super(Generator, self).__init__()

        model = [nn.ConvTranspose2d(nZ + dDis * nDis + nCon, nfG * 8, 4, 1, 0, bias=False),
                 nn.BatchNorm2d(nfG * 8),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(nfG * 8, nfG * 4, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(nfG * 4),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(nfG * 4, nfG * 2, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(nfG * 2),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(nfG * 2, nfG, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(nfG),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(nfG, nC, 4, 2, 1, bias=False),
                 nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, nC=3, nfD=64, nZ=128, nDis=10, dDis=10, nCon=0):
        super(Discriminator, self).__init__()

        self.nDis = nDis
        self.dDis = dDis
        self.nCon = nCon

        model = [nn.Conv2d(nC, nfD, 4, 2, 1),
                 # nn.BatchNorm2d(nfD), Why not to use Batchnorm here? Understand where to and where not to use Batchnorm.
                 nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfD, nfD * 2, 4, 2, 1),
                 nn.BatchNorm2d(nfD * 2),
                 nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfD * 2, nfD * 4, 4, 2, 1),
                 nn.BatchNorm2d(nfD * 4),
                 nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfD * 4, nfD * 8, 4, 2, 1),
                 nn.BatchNorm2d(nfD * 8),
                 nn.LeakyReLU(inplace=True)]

        self.commonDQ = nn.Sequential(*model)

        model = [
                 # nn.Conv2d(nfD * 4, nfD * 8, 4, 2, 1),
                 # nn.BatchNorm2d(nfD * 8),
                 # nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfD * 8, 1, 4, 1, 0), # Try a linear layer instead of conv here.
                 nn.Sigmoid()]

        self.finalD = nn.Sequential(*model)

        model = [
                 # nn.Conv2d(nfD * 4, nfD * 8, 4, 2, 1),
                 # nn.BatchNorm2d(nfD * 8),
                 # nn.LeakyReLU(inplace=True),
                 Reshape(-1, nfD * 8 * 16),
                 nn.Linear(nfD * 8 * 16, nfD * 16), # Try a conv layer instead of linear here.
                 nn.BatchNorm1d(nfD * 16),
                 nn.LeakyReLU(inplace=True),
                 nn.Linear(nfD * 16, dDis * nDis + 2 * nCon)]

        self.finalQ = nn.Sequential(*model)

    def forward(self, x):
        commonOutput = self.commonDQ(x)

        d = self.finalD(commonOutput)

        q = self.finalQ(commonOutput)
        disLogits = q[:, : self.dDis * self.nDis].squeeze()
        conMean = q[:, self.dDis * self.nDis: self.dDis * self.nDis + self.nCon].squeeze()
        conVar = F.softplus(q[:, self.dDis * self.nDis + self.nCon:].type(x.type())) + 1e-5

        return d, disLogits, conMean, conVar

    def discriminate(self, x):
        commonOutput = self.commonDQ(x)

        return self.finalD(commonOutput)

    def outputQ(self, x):
        commonOutput = self.commonDQ(x)

        q = self.finalQ(commonOutput)
        disLogits = q[:, : self.dDis * self.nDis].squeeze()
        conMean = q[:, self.dDis * self.nDis: self.dDis * self.nDis + self.nCon].squeeze()
        conVar = F.softplus(q[:, self.dDis * self.nDis + self.nCon:].type(x.type())) + 1e-5

        return disLogits, conMean, conVar

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.BatchNorm1d)):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

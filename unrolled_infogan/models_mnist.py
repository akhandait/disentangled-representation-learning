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
    def __init__(self, nZ=62, nDis=1, dDis=10, nCon=2, nC=1, nfG=64):
        super(Generator, self).__init__()

        model = [Reshape(-1, nZ + dDis * nDis + nCon),
                 nn.Linear(nZ + dDis * nDis + nCon, 1024),
                 nn.BatchNorm1d(1024),
                 nn.ReLU(inplace=True),
                 nn.Linear(1024, 7 * 7 * 128),
                 nn.BatchNorm1d(7 * 7 * 128),
                 nn.ReLU(inplace=True),
                 Reshape(-1, nfG * 2, 7, 7),
                 nn.ConvTranspose2d(nfG * 2, nfG, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(nfG),
                 nn.ReLU(inplace=True),
                 nn.ConvTranspose2d(nfG, 1, 4, 2, 1, bias=False),
                 nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, nC=1, nfD=64, nZ=62, nDis=1, dDis=10, nCon=2):
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
                 Reshape(-1, nfD * 2 * 7 * 7),
                 nn.Linear(nfD * 2 * 7 * 7, 1024),
                 nn.BatchNorm1d(1024),
                 nn.LeakyReLU(inplace=True),]

        self.commonDQ = nn.Sequential(*model)

        model = [nn.Linear(1024, 1),
                 nn.Sigmoid()]

        self.finalD = nn.Sequential(*model)

        model = [nn.Linear(1024, 128), # Try a conv layer instead of linear here.
                 nn.BatchNorm1d(128),
                 nn.LeakyReLU(inplace=True),
                 nn.Linear(128, dDis * nDis + 2 * nCon)]

        self.finalQ = nn.Sequential(*model)

    def forward(self, x):
        commonOutput = self.commonDQ(x)

        d = self.finalD(commonOutput)

        q = self.finalQ(commonOutput)
        disLogits = q[:, : self.dDis * self.nDis].squeeze()
        conMean = q[:, self.dDis * self.nDis: self.dDis * self.nDis + self.nCon].squeeze()
        conStd = F.softplus(q[:, self.dDis * self.nDis + self.nCon:].type(x.type())) + 1e-5
        # conStd = torch.exp(q[:, self.dDis * self.nDis + self.nCon:].type(x.type()))

        return d, disLogits, conMean, conStd

    def discriminate(self, x):
        commonOutput = self.commonDQ(x)

        return self.finalD(commonOutput)

    def outputQ(self, x):
        commonOutput = self.commonDQ(x)

        q = self.finalQ(commonOutput)
        disLogits = q[:, : self.dDis * self.nDis].squeeze()
        conMean = q[:, self.dDis * self.nDis: self.dDis * self.nDis + self.nCon].squeeze()
        conStd = F.softplus(q[:, self.dDis * self.nDis + self.nCon:].type(x.type())) + 1e-5
        # conStd = torch.exp(q[:, self.dDis * self.nDis + self.nCon:].type(x.type()))

        return disLogits, conMean, conStd

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.BatchNorm1d)):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

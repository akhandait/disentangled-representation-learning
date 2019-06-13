import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, inChannels=3, latentSize=256, nfE=64):
        super(Encoder, self).__init__()

        # TODO: Try BatchNorm.
        model = [nn.Conv2d(inChannels, nfE, 4, stride=2, padding=1, bias=False),
                 nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfE, nfE * 2, 4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nfE * 2, momentum=0.9),
                 nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfE * 2, nfE * 4, 4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nfE * 4, momentum=0.9),
                 nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfE * 4, nfE * 8, 4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nfE * 8, momentum=0.9),
                 nn.LeakyReLU(inplace=True),
                 Reshape(-1, nfE * 8 * 4 * 4)]

        self.LinearMean = nn.Linear(nfE * 8 * 4 * 4, latentSize)
        self.LinearStdDev = nn.Linear(nfE * 8 * 4 * 4, latentSize)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return torch.cat((self.LinearMean(x), self.LinearStdDev(x)), 1)

class Reparametrization(nn.Module):
    def __init__(self, latentSize=128, infoLatent=16):
        super(Reparametrization, self).__init__()
        self.latentSize = latentSize
        self.infoLatent = infoLatent

    def forward(self, x):
        self.mean = x[..., : self.latentSize].type(x.type())
        self.stdDeviation = nn.Softplus()(x[..., self.latentSize : ]).type(x.type()) + 1e-5

        self.sample = self.mean + self.stdDeviation * \
            torch.randn(self.mean.shape[0], self.latentSize).type(x.type())

        return self.sample

    def getInfoLatent(self):
        return self.sample[:, -self.infoLatent:]

    def klDivergence(self, mean=True):
        if not hasattr(self, 'mean'):
            raise RuntimeError('Cannot evaluate KL Divergence without a forward pass ' + \
                'before it.')

        loss = -0.5 * torch.sum(2 * torch.log(self.stdDeviation) - \
            torch.pow(self.stdDeviation, 2) - torch.pow(self.mean, 2) + 1)

        losslatent = -0.5 * torch.sum(2 * torch.log(self.stdDeviation[:, -self.infoLatent:]) - \
            torch.pow(self.stdDeviation[:, -self.infoLatent:], 2) - torch.pow(self.mean[:, -self.infoLatent:], 2) + 1) / self.infoLatent
        lossnoise = -0.5 * torch.sum(2 * torch.log(self.stdDeviation[:, :-self.infoLatent]) - \
            torch.pow(self.stdDeviation[:, :-self.infoLatent], 2) - torch.pow(self.mean[:, :-self.infoLatent], 2) + 1) \
            / (self.latentSize - self.infoLatent)

        if mean:
            return loss / self.mean.shape[0]
        return loss, losslatent, lossnoise

class Decoder(nn.Module):
    def __init__(self, outChannels=3, latentSize=128, nfD=64):
        super(Decoder, self).__init__()

        model = [nn.Linear(latentSize, nfD * 8 * 4 * 4),
                 Reshape(-1, nfD * 8, 4, 4)]

        model += [nn.ConvTranspose2d(nfD * 8, nfD * 4, 4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(nfD * 4, momentum=0.9),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(nfD * 4, nfD * 2, 4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(nfD * 2, momentum=0.9),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(nfD * 2, nfD, 4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(nfD, momentum=0.9),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(nfD, outChannels, 4, stride=2, padding=1, bias=False),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, inChannels=3, nfDis=64, infoLatent=16):
        super(Discriminator, self).__init__()

        self.infoLatent = infoLatent
        # Divide the network into two parts so that we get the features after the 3rd Convolution
        # layer to measure similarity.
        modelPartA = [nn.Conv2d(inChannels, nfDis, 4, stride=2, padding=1),
                      nn.LeakyReLU(inplace=True),
                      nn.Conv2d(nfDis, nfDis * 2, 4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(nfDis * 2, momentum=0.9),
                      nn.LeakyReLU(inplace=True),
                      nn.Conv2d(nfDis * 2, nfDis * 4, 4, stride=2, padding=1, bias=False)]

        modelPartB = [nn.BatchNorm2d(nfDis * 4, momentum=0.9),
                      nn.LeakyReLU(inplace=True),
                      nn.Conv2d(nfDis * 4, nfDis * 8, 4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(nfDis * 8, momentum=0.9),
                      nn.LeakyReLU(inplace=True)]

        modelPartC = [nn.Conv2d(nfDis * 8, 1, 4, stride=1, padding=0),
                      nn.Sigmoid()]

        modelQ = [Reshape(-1, nfDis * 8 * 16),
                  nn.Linear(nfDis * 8 * 16, nfDis * 16), # Try a conv layer instead of linear here.
                  nn.BatchNorm1d(nfDis * 16, momentum=0.9),
                  nn.LeakyReLU(inplace=True),
                  nn.Linear(nfDis * 16, 2 * infoLatent)]

        self.modelPartA = nn.Sequential(*modelPartA)
        self.modelPartB = nn.Sequential(*modelPartB)
        self.modelPartC = nn.Sequential(*modelPartC)
        self.modelQ = nn.Sequential(*modelQ)

    def forward(self, x):
        self.outPartA = self.modelPartA(x)
        outPartB = self.modelPartB(self.outPartA)

        d = self.modelPartC(outPartB).squeeze()

        q = self.modelQ(outPartB)
        meanQ = q[:, :self.infoLatent].squeeze()
        varQ = F.softplus(q[:, self.infoLatent: ].type(x.type())) + 1e-5

        return d, meanQ, varQ

    def discriminate(self, x):
        self.outPartA = self.modelPartA(x)
        outPartB = self.modelPartB(self.outPartA)

        return self.modelPartC(outPartB).squeeze()

    def outputQ(self, x):
        self.outPartA = self.modelPartA(x)
        outPartB = self.modelPartB(self.outPartA)

        q = self.modelQ(outPartB)
        meanQ = q[:, :self.infoLatent].squeeze()
        varQ = F.softplus(q[:, self.infoLatent: ].type(x.type())) + 1e-5

        return meanQ, varQ

    def getFeatures(self, x=None, savedFeatures=True):
        if savedFeatures:
            return self.outPartA.view(self.outPartA.shape[0], -1)

        features = self.modelPartA(x)
        return features.view(features.shape[0], -1)

    def clearFeatures(self):
        self.outPartA = None

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, (nn.Conv2d, nn.BatchNorm2d)):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

# Create a layer to reshape within Sequential layers, for convenience.
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


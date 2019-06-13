import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, inChannels=3, latentSize=256, nfE=64):
        super(Encoder, self).__init__()

        # TODO: Try BatchNorm.
        model = [nn.Conv2d(inChannels, nfE, 4, stride=2, padding=1),
                 nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfE, nfE * 2, 4, stride=2, padding=1),
                 nn.BatchNorm2d(nfE * 2),
                 nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfE * 2, nfE * 4, 4, stride=2, padding=1),
                 nn.BatchNorm2d(nfE * 4),
                 nn.LeakyReLU(inplace=True),
                 nn.Conv2d(nfE * 4, nfE * 8, 4, stride=2, padding=1),
                 nn.BatchNorm2d(nfE * 8),
                 nn.LeakyReLU(inplace=True)]

        model += [Reshape(-1, nfE * 8 * 4 * 4),
                  nn.Linear(nfE * 8 * 4 * 4, latentSize * 2)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Reparametrization(nn.Module):
    def __init__(self, latentSize=256):
        super(Reparametrization, self).__init__()
        self.latentSize = latentSize

    def forward(self, x):
        self.mean = x[..., : self.latentSize].type(x.type())
        self.stdDeviation = nn.Softplus()(x[..., self.latentSize : ]).type(x.type()) + 1e-5

        return self.mean + self.stdDeviation * \
            torch.randn(self.mean.shape[0], self.latentSize).type(x.type())

    def klDivergence(self, mean=True):
        if not hasattr(self, 'mean'):
            raise RuntimeError('Cannot evaluate KL Divergence without a forward pass ' + \
                'before it.')

        loss = -0.5 * torch.sum(2 * torch.log(self.stdDeviation) - \
            torch.pow(self.stdDeviation, 2) - torch.pow(self.mean, 2) + 1)

        if mean:
            return loss / self.mean.shape[0]
        return loss

class Decoder(nn.Module):
    def __init__(self, outChannels=3, latentSize=256, nfD=64):
        super(Decoder, self).__init__()

        model = [nn.Linear(latentSize, nfD * 8 * 4 * 4),
                 Reshape(-1, nfD * 8, 4, 4)]

        model += [nn.ConvTranspose2d(nfD * 8, nfD * 4, 4, stride=2, padding=1),
                  nn.BatchNorm2d(nfD * 4),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(nfD * 4, nfD * 2, 4, stride=2, padding=1),
                  nn.BatchNorm2d(nfD * 2),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(nfD * 2, nfD, 4, stride=2, padding=1),
                  nn.BatchNorm2d(nfD),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(nfD, outChannels, 4, stride=2, padding=1),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, inChannels=3, nfDis=64):
        super(Discriminator, self).__init__()

        # Divide the network into two parts so that we get the features after the 3rd Convolution
        # layer to measure similarity.
        modelPartA = [nn.Conv2d(inChannels, nfDis, 4, 2, 1),
                      nn.LeakyReLU(inplace=True),
                      nn.Conv2d(nfDis, nfDis * 2, 4, 2, 1),
                      nn.BatchNorm2d(nfDis * 2),
                      nn.LeakyReLU(inplace=True),
                      nn.Conv2d(nfDis * 2, nfDis * 4, 4, 2, 1)]

        modelPartB = [nn.BatchNorm2d(nfDis * 4),
                      nn.LeakyReLU(inplace=True),
                      nn.Conv2d(nfDis * 4, nfDis * 8, 4, 2, 1),
                      nn.BatchNorm2d(nfDis * 8),
                      nn.LeakyReLU(inplace=True),
                      nn.Conv2d(nfDis * 8, 1, 4, 1, 0),
                      nn.Sigmoid()]

        self.modelPartA = nn.Sequential(*modelPartA)
        self.modelPartB = nn.Sequential(*modelPartB)

    def forward(self, x):
        return self.modelPartB(modelPartA(x)).squeeze()

    def getFeatures(self, x):
        return F.sigmoid(modelPartA(x))

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


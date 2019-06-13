import torch.nn as nn
import torch
import numpy as np

def weightsInit(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(layer.weight.data)
    if classname.find('BatchNorm') != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

def noiseSample(nZ, nDis, dDis, nCon, batchSize):
    z = torch.randn(batchSize, nZ, 1, 1)

    ids = torch.zeros([nDis, batchSize], dtype=torch.long)
    if(nDis != 0):
        dis = torch.zeros(batchSize, nDis, dDis)

        for i in range(nDis):
            ids[i] = torch.LongTensor(batchSize).random_(0, dDis)
            dis[torch.arange(0, batchSize), i, ids[i]] = 1

        dis = dis.view(batchSize, -1, 1, 1)

    if(nCon != 0):
        # Random uniform between -1 and 1.
        con = torch.rand(batchSize, nCon, 1, 1) * 2 - 1

    noise = z
    if(nDis != 0):
        noise = torch.cat((z, dis), dim=1)
    if(nCon != 0):
        noise = torch.cat((noise, con), dim=1)

    return noise, ids


# Calculate the negative log likelihood of normal distribution.
# Treating Q(cj | x) as a factored gaussian. (Understand this.)
class NormalNLLLoss:
    def __call__(self, x, mean, std):
        logLikelihood = -0.5 * torch.log((std ** 2) * 2 * np.pi + 1e-6) - (x - mean) ** 2 / ((std ** 2) * 2 + 1e-6)
        negativeLL = -logLikelihood.sum(1).mean()

        return negativeLL

import torch.nn as nn
import torch
import numpy as np

def weightsInit(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.xavier_normal_(layer.weight.data)

    # if classname.find('BatchNorm') != -1:
    #     layer.weight.data.normal_(1.0, 0.02)
    #     layer.bias.data.fill_(0)

# Calculate the negative log likelihood of normal distribution.
# Treating Q(cj | x) as a factored gaussian. (Understand this.)
class NormalNLLLoss:
    def __call__(self, x, mean, std):
        logLikelihood = -0.5 * torch.log((std ** 2) * 2 * np.pi + 1e-6) - (x - mean) ** 2 / ((std ** 2) * 2 + 1e-6)
        # negativeLL = -logLikelihood.sum(1).mean()
        negativeLL = -logLikelihood.sum(1).sum()

        return negativeLL
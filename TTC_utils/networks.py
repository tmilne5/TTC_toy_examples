"""
Zoo of discriminator models for use with TTC
"""

import torch
from torch import nn

#################
# A 2D discriminator, for the toy examples
#################
class Discriminator(nn.Module):

    def __init__(self, dim, relu=False):
        super(Discriminator, self).__init__()

        self.nonlin = nn.ReLU() if relu else nn.Tanh()
        self.main = nn.Sequential(nn.Linear(2, dim), self.nonlin, nn.Linear(dim, 1))

    def forward(self, inputs):
        output = self.main(inputs)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torchsummary import summary
import os

class SparseAutoencoder(nn.Module):

    def __init__(self, in_dims, h_dims, sparsity_lambda=1e-3, sparsity_target=0.05, xavier_norm_init=True):
        super().__init__()
        self.in_dims = in_dims
        self.h_dims = h_dims
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target
        self.xavier_norm_init = xavier_norm_init

        """
        Map the original dimensions to a higher dimensional layer of features.
        Apply relu non-linearity to the linear transformation.
        """
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dims, self.h_dims),
            nn.Sigmoid()
        )

        if self.xavier_norm_init:
            nn.init.xavier_uniform_(self.encoder[0].weight)
            nn.init.constant_(self.encoder[0].bias, 0)

        """
        Map back the features to the original input dimensions.
        Apply relu non-linearity to the linear transformation.
        """
        self.decoder = nn.Sequential(
            nn.Linear(self.h_dims, self.in_dims),
            nn.Tanh()
        )

        if self.xavier_norm_init:
            nn.init.xavier_uniform_(self.decoder[0].weight)
            nn.init.constant_(self.decoder[0].bias, 0)

    """
    We pass the original signal through the encoder. Then we pass
    that transformation to the decoder and return both results.
    """
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def sparsity_penalty(self, encoded):
        epsilon = 1e-8
        rho_hat = torch.count_nonzero(torch.abs(encoded) < epsilon, axis=0)
        rho_hat = torch.sub(encoded.size(0), rho_hat)
        #print(rho_hat.size())
        rho = self.sparsity_target
        #epsilon = 1e-8
        # rho_hat = torch.clamp(rho_hat, min=epsilon, max=1 - epsilon)
        # print(rho_hat)
        #rho_hat = torch.count_nonzero(torch.abs(encoded) < epsilon).item()
        # print(rho_hat)
        rho_hat = torch.add(rho_hat, epsilon)
        kl_divergence = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        sparsity_penalty = torch.sum(kl_divergence)
        return self.sparsity_lambda * sparsity_penalty

    """
    Create a custom loss that combine mean squared error (MSE) loss 
    for reconstruction with the sparsity penalty.
    """
    def loss_function(self, x_hat, x, encoded):
        mse_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_penalty(encoded)
        return mse_loss + sparsity_loss

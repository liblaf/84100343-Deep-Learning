import torch
import torch.nn.functional as F
from torch import nn


class KLDivergenceLoss(nn.Module):
    def forward(self, mu, logvar):
        ############################ Your code here ############################
        # TODO: compute the KL divergence loss for q(z|x) and p(z)
        # TODO: q(z|x) ~ N(mu, exp(logvar)), p(z) ~ N(0, 1)
        ########################################################################
        # mu.shape == (batch_size, n_channels, height, width)
        kl_divergence = 0.5 * torch.sum(
            mu**2 + torch.exp(logvar) - logvar - 1, dim=(1, 2, 3)
        )
        ########################################################################
        return kl_divergence.mean()


class ReconstructionLoss(nn.Module):
    def forward(self, x, x_recon):
        return F.mse_loss(x_recon, x)


class GANLossD(nn.Module):
    def forward(self, real, fake):
        ############################ Your code here ############################
        # TODO: compute the Hinge GAN loss for the discriminator
        ########################################################################
        real = torch.mean(real, dim=(1, 2, 3))
        fake = torch.mean(fake, dim=(1, 2, 3))
        loss = 0.5 * (F.relu(1 - real) + F.relu(1 + fake))
        ########################################################################
        return loss.mean()


class GANLossG(nn.Module):
    def forward(self, fake):
        ############################ Your code here ############################
        # TODO: compute the GAN loss for the generator
        ########################################################################
        fake = torch.mean(fake, dim=(1, 2, 3))
        loss = -fake
        ########################################################################
        return loss.mean()

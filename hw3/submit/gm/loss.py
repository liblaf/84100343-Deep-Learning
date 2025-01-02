import torch
import torch.nn.functional as F
from torch import nn


class KLDivergenceLoss(nn.Module):
    def forward(self, mu, logvar):
        ############################ Your code here ############################
        # DONE: compute the KL divergence loss for q(z|x) and p(z)
        # DONE: q(z|x) ~ N(mu, exp(logvar)), p(z) ~ N(0, 1)
        ########################################################################
        # mu.shape == (batch_size, n_channels, height, width)
        kl_divergence = 0.5 * torch.sum(
            mu**2 + torch.exp(logvar) - logvar - 1, dim=tuple(range(1, mu.ndim))
        )
        ########################################################################
        return kl_divergence.mean()


class ReconstructionLoss(nn.Module):
    def forward(self, x, x_recon):
        return F.mse_loss(x_recon, x)


class GANLossD(nn.Module):
    def forward(self, real, fake):
        ############################ Your code here ############################
        # DONE: compute the Hinge GAN loss for the discriminator
        ########################################################################
        real = torch.mean(real, dim=tuple(range(1, real.ndim)))
        fake = torch.mean(fake, dim=tuple(range(1, fake.ndim)))
        loss = 0.5 * (F.relu(1 - real) + F.relu(1 + fake))
        ########################################################################
        return loss.mean()


class GANLossG(nn.Module):
    def forward(self, fake):
        ############################ Your code here ############################
        # DONE: compute the GAN loss for the generator
        ########################################################################
        fake = torch.mean(fake, dim=tuple(range(1, fake.ndim)))
        loss = -fake
        ########################################################################
        return loss.mean()

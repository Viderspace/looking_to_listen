# loss.py
import torch
import torch.nn as nn


class ComplexCompressedLoss(nn.Module):
    """
    L2 loss on power-law compressed complex spectrograms
    """

    def __init__(self):
        super(ComplexCompressedLoss, self).__init__()

    def forward(self, separated, target):
        """
        Args:
            separated: [batch, 257, 298, 2] - Separated compressed complex STFT
            target: [batch, 257, 298, 2] - Clean target compressed complex STFT

        Both are already power-law compressed!

        Returns:
            Scalar loss value
        """
        # L2 loss directly on compressed complex values
        loss = nn.functional.mse_loss(separated, target)
        return loss
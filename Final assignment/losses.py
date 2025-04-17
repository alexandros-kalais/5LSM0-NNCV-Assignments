import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W], targets: [B, H, W]
        inputs = F.softmax(inputs, dim=1)  # convert logits to probabilities

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            targets = targets * mask
            inputs = inputs * mask.unsqueeze(1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()       # [B, C, H, W]

        dims = (0, 2, 3)  # sum over batch, height, width
        intersection = torch.sum(inputs * targets_one_hot, dims)
        union = torch.sum(inputs + targets_one_hot, dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

import torch.nn as nn
import torch

class MaskWeightedMSE(nn.Module):
    def __init__(self, min_area=100):
        super(MaskWeightedMSE, self).__init__()
        self.min_area = min_area

    def forward(self, pred, label, mask):
        loss = (pred - label) ** 2
        reduce_dims = (1, 2, 3)
        delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask, dim=reduce_dims), self.min_area)
        loss = torch.sum(loss, dim=reduce_dims) / delimeter 
        loss = torch.sum(loss) / pred.size(0)
        return loss
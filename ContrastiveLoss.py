import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2)
        loss = torch.mean((1-label) * torch.pow(dist, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))

        return loss
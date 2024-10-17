import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 gamma: float = 2.0,
                 alpha: float = None,
                 reduction: str = "none",
                 label_smoothing: float = None):  
        
        super(Poly1FocalLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        if self.label_smoothing is not None:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        log_p = F.logsigmoid(logits)
        log_not_p = F.logsigmoid(-logits)

        bce_loss = -labels * log_p - (1 - labels) * log_not_p

        pt = labels * torch.exp(log_p) + (1 - labels) * torch.exp(log_not_p)
        
        pt = torch.clamp(pt, min=1e-7, max=1 - 1e-7)

        focal_term = torch.pow(1 - pt + 1e-8, self.gamma)
        FL = bce_loss * focal_term

        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1, self.gamma
    

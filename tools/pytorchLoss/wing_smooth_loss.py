import math
import torch
import torch.nn as nn

class WingLoss(nn.Module):
    def __init__(self,
                 w=10,
                 epsilon=2,
                 reduction="mean"):  # epsilon 2,1,0.5 越小越抖
        super().__init__()
        self.w, self.epsilon = w, epsilon
        self.C = w * (1.0 - math.log(1.0 + w / epsilon))
        assert reduction == 'mean' or reduction == 'sum' or reduction == 'none', "error:reduction must be [mean,sum,none]"
        self.reduction = reduction

    def forward(self, y_pred, y_true):

        x = y_true - y_pred
        absolute_x = torch.abs(x)
        loss = torch.where(self.w > absolute_x,
                           self.w * torch.log(1.0 + absolute_x / self.epsilon),
                           absolute_x - self.C)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1, reduction="mean"):  # epsilon 2,1,0.5 越小越抖
        super().__init__()
        self.beta = beta
        assert reduction == 'mean' or reduction == 'sum' or reduction == 'none', "error:reduction must be [mean,sum,none]"
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        mae = torch.abs(y_true - y_pred)
        loss = torch.where(mae > self.beta, mae - 0.5 * self.beta,
                           0.5 * mae**2 / self.beta)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

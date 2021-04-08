import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma 
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets)
        pt = torch.exp(-ce_loss) + 1e-7
        focal_loss = (self.alpha*(1-pt)**self.gamma * ce_loss)
        return self._reduce(focal_loss)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x

if __name__ == "__main__":
    criter = FocalLoss()
    criter.cuda()
    output,target = None,None
    loss = criter(output,target)

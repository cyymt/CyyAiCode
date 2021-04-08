import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    Warning: This function has no grad.
    """
    # assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))

    smooth_label = torch.empty(size=label_shape, device=true_labels.device)
    smooth_label.fill_(smoothing / (classes - 1))
    smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return smooth_label

class LabelSmoothSoftmax(nn.Module):
    """This is label smoothing loss function.
    """

    def __init__(self,lb_smooth=0.1,reduction='mean' ,dim=-1):
        super(LabelSmoothSoftmax, self).__init__()
        self.lb_smooth = lb_smooth
        self.dim = dim
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.size(1)
        true_dist = smooth_one_hot(target, num_classes, self.lb_smooth)
        if self.reduction == 'mean':
            loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        if self.reduction == 'sum':
            loss = (-true_dist * pred).reshape(-1).sum()
        return loss

class CrossEntropyLabelSmooth_other(nn.Module):
    '''
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, 0.1)
    criterion_smooth = criterion_smooth.cuda()
    '''
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth_other, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


if __name__ == "__main__":
    criter = LabelSmoothSoftmax(lb_smooth=0.1, reduction='mean')
    criter.cuda()
    output,target = None,None
    loss = criter(output,target)
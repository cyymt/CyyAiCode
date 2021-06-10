import torch
import torch.nn as nn
import torch.nn.functional as F

# relu也要二值化？micronet中使用了,todo
__all__ = ["BWNConv2d", "BWNConvTranspose2d", "BWNLinear"]


# W中心化且截断,micronet,实际加上后效果很差
def weight_center_clip(w):
    mean = w.data.mean(1, keepdim=True)
    w.data.sub_(mean)  # W中心化(C方向)
    w.data.clamp_(-1.0, 1.0)  # W截断
    return w


class BinaryWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.sign()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


class BWNConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(BWNConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        # self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001,requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # w = weight_center_clip(self.weight)
        w = self.weight
        alpha = torch.mean(torch.abs(w), (3, 2, 1), keepdim=True).detach()
        bw = BinaryWeight.apply(w) * alpha  # 输出结果是{-1,1}的矩阵

        output = F.conv2d(x, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        # import pdb; pdb.set_trace()
        return output


class BWNConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(BWNConvTranspose2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, output_padding, dilation, groups, bias,
                             padding_mode)
        # self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001,requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # w = weight_center_clip(self.weight)
        w = self.weight
        alpha = torch.mean(torch.abs(w), (3, 2, 1), keepdim=True).detach()

        bw = BinaryWeight.apply(w) * alpha

        output = F.conv_transpose2d(x, bw, self.bias, self.stride,
                                    self.padding, self.output_padding,
                                    self.groups, self.dilation)
        # import pdb; pdb.set_trace()
        return output


class BWNLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BWNLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        w = self.weight
        alpha = torch.mean(abs(w), dim=-1, keepdim=True).detach()

        bw = BinaryWeight.apply(w) * alpha

        output = F.linear(x, bw, self.bias)

        return output

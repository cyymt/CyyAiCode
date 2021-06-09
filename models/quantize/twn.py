import torch
import torch.nn as nn
import torch.nn.functional as F

# 三值网络[-1,0,1]
__all__ = ["TWNConv2d", "TWNConvTranspose2d", "TWNLinear"]


class TernaryWeight(torch.autograd.Function):
    @staticmethod
    def forward(self, input, layer_type='conv'):
        dims_size = (3, 2, 1)
        if layer_type != "conv":
            dims_size = (-1, )
        output_fp = input.clone()
        E = torch.mean(torch.abs(input), dims_size, keepdim=True)
        threshold = E * 0.7
        # ************** W —— +-1、0 **************
        output_twn = torch.sign(
            torch.add(torch.sign(torch.add(input, threshold)),
                      torch.sign(torch.add(input, -threshold))))

        # **************** α(缩放因子) ****************
        output_abs = torch.abs(output_fp)
        # <=thresh的用0代替
        filter_output = torch.where(output_abs > threshold, output_abs,
                                    torch.zeros_like(output_abs))
        alpha = torch.sum(filter_output, dims_size, keepdim=True) / torch.sum(
            output_abs.gt(threshold), dims_size, keepdim=True)
        # *************** W * α ****************
        output = output_twn * alpha  # 若不需要α(缩放因子)，注释掉即可
        return output_twn, output

    @staticmethod
    def backward(self, grad_output):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input


class TWNConv2d(nn.Conv2d):
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
        super(TWNConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        # self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001,requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        w = self.weight
        bw_twn, bw = TernaryWeight.apply(w)

        output = F.conv2d(x, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        return output


class TWNConvTranspose2d(nn.ConvTranspose2d):
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
        super(TWNConvTranspose2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, output_padding, dilation, groups, bias,
                             padding_mode)
        # self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001,requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        w = self.weight
        bw_twn, bw = TernaryWeight.apply(w)
        output = F.conv_transpose2d(x, bw, self.bias, self.stride,
                                    self.padding, self.output_padding,
                                    self.groups, self.dilation)
        return output


class TWNLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(TWNLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        w = self.weight
        bw_twn, bw = TernaryWeight.apply(w, 'fc')

        output = F.linear(x, bw, self.bias)

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DorefaConv2d", "DorefaConvTranspose2d", "DorefaLinear"]


# 第一个卷积层不能量化，因为图片是8bit，如果直接二值化，丢失信息过多
def quantize_k(r_i, nbit):
    scale = (2**nbit - 1)
    r_o = torch.round(scale * r_i) / scale
    return r_o


class DorefaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_i, nbit):
        if nbit == 32:
            output = input
        elif nbit == 1:
            print(f'{nbit} bit quantization is not supported ！')
            assert nbit != 1
        else:
            tanh = torch.tanh(r_i).float()
            # 返回的权重范围是[-1~1]
            output = 2 * quantize_k(
                tanh /
                (2 * torch.max(torch.abs(tanh)).detach()) + 0.5, nbit) - 1
        return output

    @staticmethod
    def backward(ctx, dLdr_o):
        # due to STE, dr_o / d_r_i = 1 according to formula (5)
        return dLdr_o, None


# 改善版的对激活也做量化限定
class DorefaActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, nbit):
        if nbit == 32:
            output = input
        elif nbit == 1:
            print(f'{nbit} bit quantization is not supported ！')
            assert nbit != 1
        else:
            # 对x进行截断(x截断前先进行缩放（* 0.1），目的是减小截断误差)，适应于relu激活的函数，如果是使用PACT限定的话，就不能用了
            # output = quantize_k(torch.clamp(0.1 * input, 0, 1), nbit)
            output = quantize_k(input, nbit)  # 做限制后第一层卷积对输入图片截断太多，精度差的特别厉害
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class DorefaConv2d(nn.Conv2d):
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
        super(DorefaConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        # self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001,requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, w_nbit, a_nbit):
        w = self.weight
        bw = DorefaWeight.apply(w, w_nbit)
        bx = DorefaActivation.apply(x, a_nbit)

        output = F.conv2d(bx, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        # import pdb; pdb.set_trace()
        return output


class DorefaConvTranspose2d(nn.ConvTranspose2d):
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
        super(DorefaConvTranspose2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, output_padding, dilation, groups, bias,
                             padding_mode)
        # self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001,requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, w_nbit, a_nbit):
        w = self.weight
        bw = DorefaWeight.apply(w, w_nbit)
        bx = DorefaActivation.apply(x, a_nbit)

        output = F.conv_transpose2d(bx, bw, self.bias, self.stride,
                                    self.padding, self.output_padding,
                                    self.groups, self.dilation)
        return output


class DorefaLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(DorefaLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x, w_nbit, a_nbit):
        w = self.weight
        bw = DorefaWeight.apply(w, w_nbit)
        bx = DorefaActivation.apply(x, a_nbit)

        output = F.linear(bx, bw, self.bias)

        return output

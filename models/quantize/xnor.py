import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["XnorConv2d", "XnorConvTranspose2d", "XnorLinear"]


# W中心化且截断,micronet,实际加上后效果很差
def weight_center_clip(w):
    mean = w.data.mean(1, keepdim=True)
    w.data.sub_(mean)  # W中心化(C方向)
    w.data.clamp_(-1.0, 1.0)  # W截断
    return w


# 第一个卷积层不能量化，因为图片是8bit，如果直接二值化，丢失信息过多
class BinaryActivation(torch.autograd.Function):
    '''
    Binarize the input activations for ***** BNN and XNOR *****.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.sign()  # 使用y=x函数拟合梯度
        return output

    @staticmethod
    def backward(
            ctx,
            grad_output,
    ):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # 开山之作BNN中激活值:当sign函数的输入的绝对值大于1的时候，将梯度置0可以得到更好的实验结果。
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        # 最终的梯度结果就是sign函数的梯度计算使用clip(-1,x,1)函数来拟合
        return grad_input


class BinaryWeight(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input


class XnorConv2d(nn.Conv2d):
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
        super(XnorConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        # self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001,requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # w = weight_center_clip(self.weight)
        w = self.weight
        alpha = torch.mean(torch.mean(torch.mean(abs(w), dim=3, keepdim=True),
                                      dim=2,
                                      keepdim=True),
                           dim=1,
                           keepdim=True).detach()
        bw = BinaryWeight.apply(w) * alpha  # 输出结果是{-1,1}的矩阵
        bx = BinaryActivation.apply(x)

        output = F.conv2d(bx, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        # import pdb; pdb.set_trace()
        return output


class XnorConvTranspose2d(nn.ConvTranspose2d):
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
        super(XnorConvTranspose2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, output_padding, dilation, groups, bias,
                             padding_mode)
        # self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001,requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # w = weight_center_clip(self.weight)

        alpha = torch.mean(torch.mean(torch.mean(abs(w), dim=3, keepdim=True),
                                      dim=2,
                                      keepdim=True),
                           dim=1,
                           keepdim=True).detach()

        bw = BinaryWeight.apply(w) * alpha
        bx = BinaryActivation.apply(x)

        output = F.conv_transpose2d(bx, bw, self.bias, self.stride,
                                    self.padding, self.output_padding,
                                    self.groups, self.dilation)
        # import pdb; pdb.set_trace()
        return output


class XnorLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(XnorLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        w = self.weight
        alpha = torch.mean(abs(w), dim=-1, keepdim=True).detach()

        bw = BinaryWeight.apply(w) * alpha
        bx = BinaryActivation.apply(x)

        output = F.linear(bx, bw, self.bias)

        return output

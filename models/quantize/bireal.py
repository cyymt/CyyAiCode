import torch
import torch.nn as nn
import torch.nn.functional as F

# 第一个卷积层不能量化，因为图片是8bit，如果直接二值化，丢失信息过多


class BinActiveBiReal(nn.Module):
    '''
    Binarize the input activations for ***** BiReal  *****.
    常用的激活值导数是用clip(-1,x,1)函数来代替sign函数，但在[-1,1]导数值韩式相差较大的，作者使用二阶来你和sign，缩小导数值不匹配问题
    '''
    def __init__(self):
        super(BinActiveBiReal, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        # STE反向更新函数设计
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(
            torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(
            torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(
            torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class BiRealConv2d(nn.Conv2d):
    '''
    使用时候举例子，二值和实数的嵌入
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(BasicBlock, self).__init__()


            self.binary_conv = BiRealConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(planes)

            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x # 实数信息

            # out = self.binary_activation(x)
            out = self.binary_conv(x)
            out = self.bn1(out)
            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual # 二值嵌入实数值信息

            return out
    '''
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
        super(BiRealConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        bx = BinActiveBiReal()(x)

        w = self.weight
        alpha = torch.mean(torch.mean(torch.mean(abs(w), dim=3, keepdim=True),
                                      dim=2,
                                      keepdim=True),
                           dim=1,
                           keepdim=True).detach()
        w_alpha = alpha * torch.sign(w)
        cliped_bw = torch.clamp(w, -1.0, 1.0)
        # 在训练的时候，将网络中存储的实数值的量级计入考虑，而不是只考虑了所存储的实数参数的符号
        bw = w_alpha.detach() - cliped_bw.detach(
        ) + cliped_bw  # let the gradient of binary function to 1.

        output = F.conv2d(bx, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)

        return output


class BiRealConvTranspose2d(nn.ConvTranspose2d):
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
        super(BiRealConvTranspose2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, output_padding, dilation, groups, bias,
                             padding_mode)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        bx = BinActiveBiReal()(x)

        w = self.weight
        alpha = torch.mean(torch.mean(torch.mean(abs(w), dim=3, keepdim=True),
                                      dim=2,
                                      keepdim=True),
                           dim=1,
                           keepdim=True).detach()
        w_alpha = alpha * torch.sign(w)
        cliped_bw = torch.clamp(w, -1.0, 1.0)
        # 在训练的时候，将网络中存储的实数值的量级计入考虑，而不是只考虑了所存储的实数参数的符号
        bw = w_alpha.detach() - cliped_bw.detach(
        ) + cliped_bw  # let the gradient of binary function to 1.

        output = F.conv_transpose2d(bx, bw, self.bias, self.stride,
                                    self.padding, self.output_padding,
                                    self.groups, self.dilation)

        return output


class BiRealLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BiRealLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        bx = BinActiveBiReal()(x)

        w = self.weight
        alpha = torch.mean(abs(w), dim=-1, keepdim=True).detach()
        w_alpha = alpha * torch.sign(w)
        cliped_bw = torch.clamp(w, -1.0, 1.0)
        bw = w_alpha.detach() - cliped_bw.detach(
        ) + cliped_bw  # let the gradient of binary function to 1.
        output = F.linear(bx, bw, self.bias)

        return output

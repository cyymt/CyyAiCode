#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# 第一个卷积层不能量化，因为图片是8bit，如果直接二值化，丢失信息过多

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations for ***** BNN and XNOR *****.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign() # 使用y=x函数拟合梯度
        return input
    @staticmethod
    def backward(ctx, grad_output, ):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # 开山之作BNN中激活值:当sign函数的输入的绝对值大于1的时候，将梯度置0可以得到更好的实验结果。
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        # 最终的梯度结果就是sign函数的梯度计算使用clip(-1,x,1)函数来拟合
        return grad_input


class BNNConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1,bias=True,padding_mode='zeros'):
        super(BNNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,dilation,groups,bias,padding_mode)
        # self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001,requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        w = self.weight

        bw = BinActive().apply(w) # 输出结果是{-1,1}的矩阵
        bx = BinActive().apply(x)

        output = F.conv2d(bx, bw, self.bias, self.stride, self.padding,self.dilation,self.groups)
        # import pdb; pdb.set_trace()
        return output


class BNNLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(BNNLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        w = self.weight

        bw = BinActive().apply(w)
        bx = BinActive().apply(x)

        output = F.linear(bx, bw, self.bias)

        return output

class BWNConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1,bias=True,padding_mode='zeros'):
        super(BWNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,dilation,groups,bias,padding_mode)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):

        w = self.weight
        alpha = torch.mean(
            torch.mean(torch.mean(abs(w), dim=3, keepdim=True), dim=2, keepdim=True), dim=1,
            keepdim=True).detach()
        bw = BinActive().apply(w)
        bw = bw * alpha
        output = F.conv2d(x, bw, self.bias, self.stride, self.padding,self.dilation,self.groups)

        return output

class BWNLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(BWNLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        w = self.weight
        alpha = torch.mean(abs(w), dim=-1, keepdim=True).detach()

        bw = BinActive().apply(w)
        bw = bw * alpha 

        output = F.linear(x, bw, self.bias)

        return output

class XnorConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1,bias=True,padding_mode='zeros'):
        super(XnorConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,dilation,groups,bias,padding_mode)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):

        w = self.weight
        alpha = torch.mean(
            torch.mean(torch.mean(abs(w), dim=3, keepdim=True), dim=2, keepdim=True), dim=1,
            keepdim=True).detach()
        bw = BinActive().apply(w)
        bx = BinActive().apply(x)
        bw = bw * alpha
        output = F.conv2d(bx, bw, self.bias, self.stride, self.padding,self.dilation,self.groups)
        return output

class XnorLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(XnorLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        w = self.weight
        alpha = torch.mean(abs(w), dim=-1, keepdim=True).detach()

        bw = BinActive().apply(w)
        bw = bw * alpha 
        bx = BinActive().apply(x)

        output = F.linear(bx, bw, self.bias)

        return output


class BinActiveBiReal(nn.Module):
    '''
    Binarize the input activations for ***** BiReal  *****.
    常用的激活值导数是用clip(-1,x,1)函数来代替sign函数，但在[-1,1]导数值韩式相差较大的，作者使用二阶来你和sign，缩小导数值不匹配问题
    '''
    def __init__(self):
        super(BinActiveBiReal, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)

        # STE反向更新函数设计
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1,bias=True,padding_mode='zeros'):
        super(BiRealConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,dilation,groups,bias,padding_mode)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        bx = BinActiveBiReal()(x)

        w = self.weight
        alpha = torch.mean(
            torch.mean(torch.mean(abs(w), dim=3, keepdim=True), dim=2, keepdim=True), dim=1,
            keepdim=True).detach()
        w_alpha = alpha * torch.sign(w)
        cliped_bw = torch.clamp(w, -1.0, 1.0)
        # 在训练的时候，将网络中存储的实数值的量级计入考虑，而不是只考虑了所存储的实数参数的符号
        bw = w_alpha.detach() - cliped_bw.detach() + cliped_bw  # let the gradient of binary function to 1.
        
        output = F.conv2d(bx, bw, self.bias, self.stride, self.padding,self.dilation,self.groups)

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
        bw = w_alpha.detach() - cliped_bw.detach() + cliped_bw  # let the gradient of binary function to 1.
        output = F.linear(bx, bw, self.bias)
        
        return output
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DorefaConv2d", "DorefaConvTranspose2d", "DorefaLinear"]

# github:https://github.com/KwangHoonAn/PACT # 如何使用见代码，self.alpha写在__init__是必须的，因为是要优化的参数
# PACT_SRC=ReluK,所有的relu替换为PACT,注意:代码中第一层和最后一层都是8bit,其他4bit:
# k = 8,alpha1 = nn.Parameter(torch.tensor(10.))
# out = F.relu(self.bn1(self.conv1(x)))
# out = PACT_SRC.apply(self.bn1(self.conv1(x)), self.alpha1, K)


# alpha参数如何更新？
# loss = criterion(output, target_var)
# l2_alpha = 0.0
# for name, param in model.named_parameters():
#     if "alpha" in name: # 参数写在__init__中，是一个属性，故可以调用优化
#         l2_alpha += torch.pow(param, 2) # L2 regularization
# loss += lambda_alpha * l2_alpha # lambda_alpha = 0.0002
class PACT_SRC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, k):
        ctx.save_for_backward(x, alpha)
        # y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
        y = torch.clamp(x, min=0, max=alpha.item())
        # 下面的就是dorafa的激活量化，所以：如果dorefa+pact的话，dorefa中的激活量化可以不要
        # 为什么除以alpha，因为输入值范围到[0,1](同dorefa激活clamp(0.1*input,0,1)),clamp之后的范围是[0,alpha],所以才除以alpha
        scale = (2**k - 1) / alpha
        y_q = torch.round(y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        # Backward function, I borrowed code from
        # https://github.com/obilaniu/GradOverride/blob/master/functional.py
        # We get dLoss / dy_q as a gradient
        x, alpha, = ctx.saved_tensors
        # Weight gradient is only valid when [0, alpha]
        # Actual gradient for alpha,
        # By applying Chain Rule, we get (dLoss / dy_q) * (dy_q / dy) * (dy / dalpha)
        # dLoss / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range
        lower_bound = x < 0
        upper_bound = x > alpha
        # x_range       = 1.0-lower_bound-upper_bound
        x_range = ~(lower_bound | upper_bound)  # True False矩阵，在范围内的为True
        # ge:>=，grad_alpha:计算大于等于alpha的梯度和作为参数alpha的梯度
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
        # # backward的输出个数，应与forward的输入个数相同,如果不需要梯度，返回None即可，k不需要梯度
        return dLdy_q * x_range.float(), grad_alpha, None


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
            # 1.做限制后第一层卷积对输入图片截断太多，精度差的特别厉害
            # 2.使用改用线性量化后
            output = quantize_k(input, nbit)
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

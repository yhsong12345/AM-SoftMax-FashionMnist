import torch
import torch.nn as nn



def autopad(kernel_size, p):
    if p == None:
        p = kernel_size //2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
    return p



class conv(nn.Module):
    def __init__(self, input, output, kernel_size=3, stride=1, p=None):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(input, output, kernel_size, stride, 
                              padding=autopad(kernel_size, p), bias = False)
        self.bn = nn.BatchNorm2d(output)
        self.act = nn.ReLU(inplace=True)
        self.max = nn.MaxPool2d(2)

    
    def forward(self, x):
        return self.max(self.act(self.bn(self.conv(x))))
    

    

# class Anglelayer(nn.Module):
#     def __init__(self, input_channel, output_channel, m=4):
#         super(Anglelayer, self).__init__()
#         self.input_channel = input_channel
#         self.output_channel = output_channel
#         self.m = m
#         self.weight = nn.Parameter(torch.Tensor(self.input_channel, self.output_channel))
#         self.weight.data.uniform_(-1,1).renorm_(2, 1, 1e-5).mul_(1e5)
#         self.cos_val = [
#             lambda x: x**0,
#             lambda x: x**1,
#             lambda x: 2*x**2-1,
#             lambda x: 4*x**3-3*x,
#             lambda x: 8*x**4-8*x**2+1,
#             lambda x: 16*x**5-20*x**3+5*x,
#         ]
    
#     def forward(self, input):
#         w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
#         x_modulus = input.pow(2).sum(1).pow(0.5)
#         w_modulus = w.pow(2).sum(0).pow(0.5)

#         # W * x = ||W|| * ||x|| * cos(θ)
#         inner_wx = input.mm(w)
#         cos_theta = inner_wx / x_modulus.view(-1, 1) / w_modulus.view(1, -1)
#         cos_theta = cos_theta.clamp(-1, 1)

#         cos_m_theta = self.cos_val[self.m](cos_theta)
#         theta = torch.autograd.Variable(cos_theta.data.acos())
#         # k * pi / m <= theta <= (k + 1) * pi / m
#         k = (self.m * theta / 3.14159265).floor()
#         minus_one = k * 0.0 - 1
#         # Phi(yi, i) = (-1)**k * cos(myi,i) - 2 * k
#         phi_theta = (minus_one ** k) * cos_m_theta - 2 * k

#         cos_x = cos_theta * x_modulus.view(-1, 1)
#         phi_x = phi_theta * x_modulus.view(-1, 1)

#         return cos_x, phi_x
        
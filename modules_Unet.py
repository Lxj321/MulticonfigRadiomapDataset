# -*- coding: utf-8 -*-
"""
修改后的modules_updated.py - 支持beam参数注入

包含两种方案：
1. 连续值编码：直接增加4个输入通道
2. FiLM调制：保持3个输入通道，通过FiLM层注入参数
"""
import torch.nn as nn
import torch
import functools

# ==================== 原始ResnetBlock（方案1使用） ====================

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# ==================== FiLM相关组件（方案2使用） ====================

class FiLMGenerator(nn.Module):
    """FiLM参数生成器"""
    def __init__(self, param_dim=4, hidden_dim=128, num_film_layers=3, channel_dims=[64, 128, 256]):
        super().__init__()
        self.num_layers = num_film_layers
        
        # 参数编码器
        self.encoder = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True)
        )
        
        # 为每一层生成FiLM参数
        self.film_layers = nn.ModuleList()
        for channels in channel_dims:
            self.film_layers.append(
                nn.Linear(hidden_dim, channels * 2)
            )
    
    def forward(self, beam_params):
        """
        beam_params: (B, 4)
        返回: [(gamma_1, beta_1), (gamma_2, beta_2), ...]
        """
        h = self.encoder(beam_params)
        
        film_params = []
        for film_layer in self.film_layers:
            params = film_layer(h)
            channels = params.size(1) // 2
            gamma = params[:, :channels]
            beta = params[:, channels:]
            film_params.append((gamma, beta))
        
        return film_params


class FiLMResnetBlock(nn.Module):
    """带FiLM调制的ResNet块"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]
        
        return nn.Sequential(*conv_block)
    
    def forward(self, x, gamma=None, beta=None):
        out = self.conv_block(x)
        
        if gamma is not None and beta is not None:
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
            out = gamma * out + beta
        
        out = out + x
        return out


# ==================== 方案1: 原始生成器（支持更多输入通道） ====================

class ResnetGenerator(nn.Module):
    """
    方案1专用：连续值编码
    
    使用方法：
    model = ResnetGenerator(
        input_nc=7,  # 2+1+4 = 建筑物+Tx+3D高度+4维beam参数
        output_nc=1
    )
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=3, gpu_id='cuda:0', padding_type='zero'):
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_id
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        # 编码器
        self.inpl = nn.ZeroPad2d(3)
        self.conv1 = nn.Conv2d(self.input_nc, self.ngf, kernel_size=7, padding=0, bias=use_bias)
        self.norm1 = norm_layer(ngf)
        self.r1 = nn.ReLU(True)
        
        ds = 2
        self.conv2 = nn.Conv2d(self.ngf, self.ngf*ds, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.norm2 = norm_layer(ngf*ds)
        self.r2 = nn.ReLU(True)
        
        self.conv3 = nn.Conv2d(self.ngf*ds, self.ngf*ds**2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.norm3 = norm_layer(ngf*ds**2)
        self.r3 = nn.ReLU(True)
        
        # ResNet块
        factor = 2**ds
        self.resnet1 = ResnetBlock(self.ngf*factor, padding_type=padding_type, 
                                   norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.resnet2 = ResnetBlock(self.ngf*factor, padding_type=padding_type, 
                                   norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        
        # 解码器
        uf1 = 2**ds
        uf2 = 2**(ds-1)
        self.upconv1 = nn.ConvTranspose2d(ngf*uf1, int(ngf*uf1/2), kernel_size=3, 
                                         stride=2, padding=1, output_padding=1, bias=use_bias)
        self.upnorm1 = norm_layer(int(ngf*uf1/2))
        self.upr1 = nn.ReLU(True)
        
        self.upconv2 = nn.ConvTranspose2d(ngf*uf2, int(ngf*uf2/2), kernel_size=3, 
                                         stride=2, padding=1, output_padding=1, bias=use_bias)
        self.upnorm2 = norm_layer(int(ngf*uf2/2))
        self.upr2 = nn.ReLU(True)
        
        self.uppad1 = nn.ReplicationPad2d(3)
        self.convf = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        self.tan = nn.Tanh()
    
    def forward(self, x):
        x = self.r1(self.norm1(self.conv1(self.inpl(x))))
        x = self.r2(self.norm2(self.conv2(x)))
        x = self.r3(self.norm3(self.conv3(x)))
        
        f1 = self.resnet1(x)
        f6 = self.resnet2(f1)
        
        y = self.upr1(self.upnorm1(self.upconv1(f6)))
        y = self.upr2(self.upnorm2(self.upconv2(y)))
        y = self.tan(self.convf(self.uppad1(y)))
        
        return y


# ==================== 方案2: FiLM生成器（推荐！） ====================

class FiLMResnetGenerator(nn.Module):
    """
    方案2专用：FiLM调制
    
    使用方法：
    model = FiLMResnetGenerator(
        input_nc=3,  # 仅3通道！建筑物+Tx+3D高度
        output_nc=1,
        use_film=True
    )
    
    # 前向传播
    output = model(images, beam_params)  # 需要额外传入beam参数
    """
    def __init__(self, input_nc=3, output_nc=1, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=2, padding_type='zero', 
                 use_film=True, param_dim=4):
        super().__init__()
        
        self.use_film = use_film
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        # FiLM生成器
        if self.use_film:
            self.film_generator = FiLMGenerator(
                param_dim=param_dim, 
                hidden_dim=128, 
                num_film_layers=3,
                channel_dims=[ngf*2, ngf*4, ngf*4]
            )
        
        # 编码器
        self.inpl = nn.ZeroPad2d(3)
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)
        self.norm1 = norm_layer(ngf)
        self.r1 = nn.ReLU(True)
        
        ds = 2
        self.conv2 = nn.Conv2d(ngf, ngf*ds, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.norm2 = norm_layer(ngf*ds)
        self.r2 = nn.ReLU(True)
        
        self.conv3 = nn.Conv2d(ngf*ds, ngf*ds**2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.norm3 = norm_layer(ngf*ds**2)
        self.r3 = nn.ReLU(True)
        
        # ResNet块（带FiLM）
        factor = 2**ds
        self.resnet1 = FiLMResnetBlock(ngf*factor, padding_type, norm_layer, use_dropout, use_bias)
        self.resnet2 = FiLMResnetBlock(ngf*factor, padding_type, norm_layer, use_dropout, use_bias)
        
        # 解码器
        uf1 = 2**ds
        uf2 = 2**(ds-1)
        self.upconv1 = nn.ConvTranspose2d(ngf*uf1, int(ngf*uf1/2), 
                                         kernel_size=3, stride=2, padding=1, 
                                         output_padding=1, bias=use_bias)
        self.upnorm1 = norm_layer(int(ngf*uf1/2))
        self.upr1 = nn.ReLU(True)
        
        self.upconv2 = nn.ConvTranspose2d(ngf*uf2, int(ngf*uf2/2), 
                                         kernel_size=3, stride=2, padding=1, 
                                         output_padding=1, bias=use_bias)
        self.upnorm2 = norm_layer(int(ngf*uf2/2))
        self.upr2 = nn.ReLU(True)
        
        self.uppad1 = nn.ReplicationPad2d(3)
        self.convf = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        self.tan = nn.Tanh()
    
    def forward(self, x, beam_params=None):
        """
        x: (B, 3, H, W)
        beam_params: (B, 4) - [freq, tr, num_beams, beam_id] (归一化后)
        """
        # 生成FiLM参数
        film_params = None
        if self.use_film and beam_params is not None:
            film_params = self.film_generator(beam_params)
        
        # 编码
        x = self.r1(self.norm1(self.conv1(self.inpl(x))))
        
        # 下采样 + FiLM调制
        x = self.r2(self.norm2(self.conv2(x)))
        if film_params:
            gamma1, beta1 = film_params[0]
            gamma1 = gamma1.unsqueeze(2).unsqueeze(3)
            beta1 = beta1.unsqueeze(2).unsqueeze(3)
            x = gamma1 * x + beta1
        
        x = self.r3(self.norm3(self.conv3(x)))
        if film_params:
            gamma2, beta2 = film_params[1]
            gamma2 = gamma2.unsqueeze(2).unsqueeze(3)
            beta2 = beta2.unsqueeze(2).unsqueeze(3)
            x = gamma2 * x + beta2
        
        # ResNet块 + FiLM
        gamma3, beta3 = film_params[2] if film_params else (None, None)
        f1 = self.resnet1(x, gamma3, beta3)
        f6 = self.resnet2(f1, gamma3, beta3)
        
        # 解码
        y = self.upr1(self.upnorm1(self.upconv1(f6)))
        y = self.upr2(self.upnorm2(self.upconv2(y)))
        y = self.tan(self.convf(self.uppad1(y)))
        
        return y


# ==================== RadioWNet (W型网络) ====================

class RadioWNet(nn.Module):
    def __init__(self, inputs=3, phase="firstU", use_film=True):
        super().__init__()
        self.phase = phase
        self.use_film = use_film
        
        if use_film:
            # 第一个U-Net: 输入channels不变
            self.firstU = FiLMResnetGenerator(
                input_nc=inputs, 
                output_nc=1, 
                use_film=True
            )
            # 第二个U-Net: 输入channels = inputs + 1 (加上第一个U-Net的输出)
            self.secondU = FiLMResnetGenerator(
                input_nc=inputs + 1,  # ✅ 关键修改!
                output_nc=1, 
                use_film=True
            )
        else:
            self.firstU = ResnetGenerator(
                input_nc=inputs, 
                output_nc=1
            )
            self.secondU = ResnetGenerator(
                input_nc=inputs + 1,  # ✅ 关键修改!
                output_nc=1
            )
        
        # 冻结策略
        if phase == "firstU":
            for param in self.secondU.parameters():
                param.requires_grad = False
        elif phase == "secondU":
            for param in self.firstU.parameters():
                param.requires_grad = False
    
    def forward(self, x, beam_params=None):
        """
        x: 输入图像
        beam_params: beam参数（仅FiLM模式需要）
        """
        if self.use_film:
            # 第一个U-Net
            if self.phase == "firstU":
                output1 = self.firstU(x, beam_params)
                # 拼接第一个U-Net的输出和原始输入
                x_with_output1 = torch.cat([output1.detach(), x], dim=1)
                output2 = self.secondU(x_with_output1, beam_params).detach()
            else:  # secondU阶段
                output1 = self.firstU(x, beam_params).detach()
                # 拼接第一个U-Net的输出和原始输入
                x_with_output1 = torch.cat([output1, x], dim=1)
                output2 = self.secondU(x_with_output1, beam_params)
        else:
            if self.phase == "firstU":
                output1 = self.firstU(x)
                x_with_output1 = torch.cat([output1.detach(), x], dim=1)
                output2 = self.secondU(x_with_output1).detach()
            else:
                output1 = self.firstU(x).detach()
                x_with_output1 = torch.cat([output1, x], dim=1)
                output2 = self.secondU(x_with_output1)
        
        return [output1, output2]


# class RadioWNet(nn.Module):
#     """
#     W型网络 - 支持两种方案
    
#     方案1用法：
#     model = RadioWNet(inputs=7, phase="firstU", use_film=False)
#     output1, output2 = model(x)
    
#     方案2用法（推荐）：
#     model = RadioWNet(inputs=3, phase="firstU", use_film=True)
#     output1, output2 = model(x, beam_params)
#     """
#     def __init__(self, inputs=3, phase="firstU", use_film=True):
#         super().__init__()
#         self.phase = phase
#         self.use_film = use_film
        
#         if use_film:
#             # 方案2: FiLM调制
#             self.firstU = FiLMResnetGenerator(
#                 input_nc=inputs, 
#                 output_nc=1, 
#                 use_film=True
#             )
#             self.secondU = FiLMResnetGenerator(
#                 input_nc=inputs, 
#                 output_nc=1, 
#                 use_film=True
#             )
#         else:
#             # 方案1: 连续值编码
#             self.firstU = ResnetGenerator(
#                 input_nc=inputs, 
#                 output_nc=1
#             )
#             self.secondU = ResnetGenerator(
#                 input_nc=inputs, 
#                 output_nc=1
#             )
        
#         # 冻结策略
#         if phase == "firstU":
#             for param in self.secondU.parameters():
#                 param.requires_grad = False
#         elif phase == "secondU":
#             for param in self.firstU.parameters():
#                 param.requires_grad = False
    
#     def forward(self, x, beam_params=None):
#         """
#         x: 输入图像
#         beam_params: beam参数（仅FiLM模式需要）
#         """
#         if self.use_film:
#             output1 = self.firstU(x, beam_params)
#             output2 = self.secondU(x, beam_params)
#         else:
#             output1 = self.firstU(x)
#             output2 = self.secondU(x)
        
#         return [output1, output2]


class Discriminator(nn.Module):
    """判别器（保持不变）"""
    def __init__(self, ngpu, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

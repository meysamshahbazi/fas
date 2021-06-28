import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import namedtuple
import math
from inspect import isfunction



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class GDC(nn.Module):
    def __init__(self, in_c, emb_size):
        super(GDC, self).__init__()
        self.conv_6_dw = Linear_block(in_c, in_c, groups=in_c, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(in_c, emb_size, bias=False)
        #self.bn = BatchNorm1d(emb_size, affine=False)
        self.bn = nn.BatchNorm1d(emb_size)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.avgpool(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x






def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.PReLU(out_planes) #nn.ReLU6(inplace=True)
        )


class SandGlass(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, identity_tensor_multiplier=1.0, norm_layer=None):
        super(SandGlass, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_identity = False if identity_tensor_multiplier==1.0 else True
        self.identity_tensor_channels = int(round(inp*identity_tensor_multiplier))

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp / expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        # dw
        layers.append(ConvBNReLU(inp, inp, kernel_size=3, stride=1, groups=inp, norm_layer=norm_layer))
        if expand_ratio != 1:
            # pw-linear
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                norm_layer(hidden_dim),
            ])
        layers.extend([
            # pw
            ConvBNReLU(hidden_dim, oup, kernel_size=1, stride=1, groups=1, norm_layer=norm_layer),
            # dw-linear
            nn.Conv2d(oup, oup, kernel_size=3, stride=stride, groups=oup, padding=1, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.use_identity:
                identity_tensor= x[:,:self.identity_tensor_channels,:,:] + out[:,:self.identity_tensor_channels,:,:]
                out = torch.cat([identity_tensor, out[:,self.identity_tensor_channels:,:,:]], dim=1)
                # out[:,:self.identity_tensor_channels,:,:] += x[:,:self.identity_tensor_channels,:,:]
            else:
                out = x + out
            return out
        else:
            return out
class ECA_Layer(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super(ECA_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    """
    def __init__(self, channels, reduction=16, round_mid=False, use_conv=True,
                 mid_activation=(lambda: nn.ReLU(inplace=True)),
                 out_activation=(lambda: nn.Sigmoid())):
        super(SEBlock, self).__init__()
        self.use_conv = use_conv
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1,
                                stride=1, groups=1, bias=True)
        else:
            self.fc1 = nn.Linear(in_features=channels, out_features=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        if use_conv:
            self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1,
                                stride=1, groups=1, bias=True)
        else:
            self.fc2 = nn.Linear(in_features=mid_channels, out_features=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x

class Identity(nn.Module):
    """
    Identity block.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GCT(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate




class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.PReLU(out_planes) #nn.ReLU6(inplace=True)
        )


class InvertedResidual_V2(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual_V2, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup)
    )

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(nn.Module):
     def __init__(self, in_c, out_c, attention, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.attention = attention
        if self.attention == 'eca':
            self.attention_layer = ECA_Layer(out_c)
        elif self.attention == 'se':
            self.attention_layer = SEBlock(out_c)
        elif self.attention == 'cbam':
            self.attention_layer = CbamBlock(out_c)
        elif self.attention == 'gct':
            self.attention_layer = GCT(out_c)

        self.residual = residual

        self.attention = attention #se, eca, cbam
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.attention != 'none':
            x = self.attention_layer(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(nn.Module):
    def __init__(self, c, attention, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, attention, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class GNAP(nn.Module):
    def __init__(self, emb_size):
        super(GNAP, self).__init__()
        assert emb_size == 512
        self.bn1 = BatchNorm2d(512, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = BatchNorm1d(512, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature





class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.PReLU(hidden_dim), #if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.PReLU(hidden_dim),# if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                nn.PReLU(hidden_dim),# if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 input_size, emb_size = 512, nb_ch=3,
                 width_mult=1.0, inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()
        #assert input_size[0] in [112]
        if block is None:
            block = InvertedResidual_V2

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 512

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                #[6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(nb_ch, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.output_layer = GDC(512, emb_size)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.output_layer(x)
        return x



large_cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
small_cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]
class MobileNetV3(nn.Module):
    def __init__(self, input_size, emb_size = 512,nb_ch=3, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        #assert input_size[0] in [112]
        self.cfgs = large_cfgs
    
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(nb_ch, input_channel, 1)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, 512)
        '''
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )
        '''
        self.output_layer = GDC(512, emb_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.output_layer(x)
        return x


class MobileNeXt(nn.Module):
    def __init__(self,
                input_size, emb_size = 512,nb_ch=3,
                width_mult=1.0, identity_tensor_multiplier=1.0,
                sand_glass_setting=None,
                round_nearest=8,
                block=None,
                norm_layer=None):
        """
        MobileNeXt main class
        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            sand_glass_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNeXt, self).__init__()
        #assert input_size[0] in [112]
        if block is None:
            block = SandGlass

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 512

        if sand_glass_setting is None:
            sand_glass_setting = [
                # t, c,  b, s
                #[2, 96,  1, 2],
                [6, 144, 1, 1],
                [6, 192, 3, 2],
                [6, 288, 3, 2],
                [6, 384, 4, 1],
                [6, 576, 4, 2],
                [6, 960, 2, 1],
                [6, last_channel, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(sand_glass_setting) == 0 or len(sand_glass_setting[0]) != 4:
            raise ValueError("sand_glass_setting should be non-empty "
                             "or a 4-element list, got {}".format(sand_glass_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(nb_ch, input_channel, stride=2, norm_layer=norm_layer)]
        # building sand glass blocks
        for t, c, b, s in sand_glass_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(b):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, identity_tensor_multiplier=identity_tensor_multiplier, norm_layer=norm_layer))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.output_layer = GDC(512, emb_size)
        

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.output_layer(x)
        return x


class MobileFaceNet(nn.Module):
    def __init__(self, input_size, emb_size = 512,nb_ch=3, output_name = "GDC", attention='none'):
        super(MobileFaceNet, self).__init__()
        assert output_name in ["GNAP", 'GDC']
        #assert input_size[0] in [112]

        self.conv1 = Conv_block(nb_ch, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, attention, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, attention, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, attention, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, attention, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, attention, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, attention, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        if output_name == "GNAP":
            self.output_layer = GNAP(emb_size)
        else:
            self.output_layer = GDC(512, emb_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)

        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        conv_features = self.conv_6_sep(out)
        out = self.output_layer(conv_features)
        return out


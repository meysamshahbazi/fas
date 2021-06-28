import os
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch


def round_channels(channels, divisor=8):
    """
    Round weighted channel number (make divisible operation).

    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.

    Returns
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels
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
        self.conv_6_flatten = nn.Flatten()
        self.linear = nn.Linear(in_c, emb_size, bias=False)
        #self.bn = BatchNorm1d(emb_size, affine=False)
        self.bn = nn.BatchNorm1d(emb_size)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


def get_activation_layer(activation, out_channels):
    """
    Create activation layer from string/function.
    """
    assert (activation is not None)

    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "relu6":
        return nn.ReLU6(inplace=True)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "prelu":
        return nn.PReLU(out_channels)
    else:
        raise NotImplementedError()


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.
    """
    def __init__(self, nb_ch, out_channels, kernel_size, stride, padding,
                dilation=1, groups=1, bias=False, use_bn=True, bn_eps=1e-5,
                activation=(lambda: nn.ReLU(inplace=True))):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0
        self.conv = nn.Conv2d(in_channels=nb_ch, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        if self.activate:
            self.activ = get_activation_layer(activation, out_channels)

    def forward(self, x):
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(nb_ch, out_channels, stride=1, padding=0, groups=1,
                bias=False, use_bn=True, bn_eps=1e-5,
                activation=(lambda: nn.ReLU(inplace=True))):
    """
    1x1 version of the standard convolution block.
    """
    return ConvBlock(nb_ch=nb_ch, out_channels=out_channels, kernel_size=1, stride=stride,
        padding=padding, groups=groups, bias=bias, use_bn=use_bn, bn_eps=bn_eps, activation=activation)

def conv3x3_block(nb_ch, out_channels, stride=1, padding=1, dilation=1, groups=1,
                bias=False, use_bn=True, bn_eps=1e-5, activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 version of the standard convolution block.
    """
    return ConvBlock(nb_ch=nb_ch, out_channels=out_channels, kernel_size=3, stride=stride,
        padding=padding, dilation=dilation, groups=groups, bias=bias, use_bn=use_bn, bn_eps=bn_eps,
        activation=activation)

def dwconv_block(nb_ch, out_channels, kernel_size, stride=1, padding=1, dilation=1,
                bias=False, use_bn=True, bn_eps=1e-5, activation=(lambda: nn.ReLU(inplace=True))):
    """
    Depthwise version of the standard convolution block.
    """
    return ConvBlock(nb_ch=nb_ch, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, groups=out_channels, bias=bias, use_bn=use_bn, bn_eps=bn_eps, activation=activation)

def dwconv3x3_block(nb_ch, out_channels, stride=1, padding=1, dilation=1,
                    bias=False, bn_eps=1e-5, activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 depthwise version of the standard convolution block.
    """
    return dwconv_block(nb_ch=nb_ch, out_channels=out_channels, kernel_size=3, stride=stride,
        padding=padding, dilation=dilation, bias=bias, bn_eps=bn_eps, activation=activation)

def dwconv5x5_block(nb_ch, out_channels, stride=1, padding=2, dilation=1,
                    bias=False, bn_eps=1e-5, activation=(lambda: nn.ReLU(inplace=True))):
    """
    5x5 depthwise version of the standard convolution block.
    """
    return dwconv_block( nb_ch=nb_ch, out_channels=out_channels, kernel_size=5, stride=stride,
        padding=padding, dilation=dilation, bias=bias, bn_eps=bn_eps, activation=activation)

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



def calc_tf_padding(x,
                    kernel_size,
                    stride=1,
                    dilation=1):
    """
    Calculate TF-same like padding size.

    Parameters:
    ----------
    x : tensor
        Input tensor.
    kernel_size : int
        Convolution window size.
    stride : int, default 1
        Strides of the convolution.
    dilation : int, default 1
        Dilation value for convolution layer.

    Returns
    -------
    tuple of 4 int
        The size of the padding.
    """
    height, width = x.size()[2:]
    oh = math.ceil(height / stride)
    ow = math.ceil(width / stride)
    pad_h = max((oh - 1) * stride + (kernel_size - 1) * dilation + 1 - height, 0)
    pad_w = max((ow - 1) * stride + (kernel_size - 1) * dilation + 1 - width, 0)
    return pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2


class EffiDwsConvUnit(nn.Module):
    """
    EfficientNet specific depthwise separable convolution block/unit with BatchNorms and activations at each convolution
    layers.

    Parameters:
    ----------
    nb_ch : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    bn_eps : float
        Small float added to variance in Batch norm.
    activation : str
        Name of activation function.
    tf_mode : bool
        Whether to use TF-like mode.
    """
    def __init__(self,
                 nb_ch,
                 out_channels,
                 stride,
                 bn_eps,
                 activation,
                 tf_mode):
        super(EffiDwsConvUnit, self).__init__()
        self.tf_mode = tf_mode
        self.residual = (nb_ch == out_channels) and (stride == 1)

        self.dw_conv = dwconv3x3_block(
            nb_ch=nb_ch,
            out_channels=nb_ch,
            padding=(0 if tf_mode else 1),
            bn_eps=bn_eps,
            activation=activation)
        self.se = SEBlock(
            channels=nb_ch,
            reduction=4,
            mid_activation=activation)
        self.pw_conv = conv1x1_block(
            nb_ch=nb_ch,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=3))
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.pw_conv(x)
        if self.residual:
            x = x + identity
        return x


class EffiInvResUnit(nn.Module):
    """
    EfficientNet inverted residual unit.

    Parameters:
    ----------
    nb_ch : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    exp_factor : int
        Factor for expansion of channels.
    se_factor : int
        SE reduction factor for each unit.
    bn_eps : float
        Small float added to variance in Batch norm.
    activation : str
        Name of activation function.
    tf_mode : bool
        Whether to use TF-like mode.
    """
    def __init__(self,
                 nb_ch,
                 out_channels,
                 kernel_size,
                 stride,
                 exp_factor,
                 se_factor,
                 bn_eps,
                 activation,
                 tf_mode):
        super(EffiInvResUnit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.tf_mode = tf_mode
        self.residual = (nb_ch == out_channels) and (stride == 1)
        self.use_se = se_factor > 0
        mid_channels = nb_ch * exp_factor
        dwconv_block_fn = dwconv3x3_block if kernel_size == 3 else (dwconv5x5_block if kernel_size == 5 else None)

        self.conv1 = conv1x1_block(
            nb_ch=nb_ch,
            out_channels=mid_channels,
            bn_eps=bn_eps,
            activation=activation)
        self.conv2 = dwconv_block_fn(
            nb_ch=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            padding=(0 if tf_mode else (kernel_size // 2)),
            bn_eps=bn_eps,
            activation=activation)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=(exp_factor * se_factor),
                mid_activation=activation)
        self.conv3 = conv1x1_block(
            nb_ch=mid_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=self.kernel_size, stride=self.stride))
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class EffiInitBlock(nn.Module):
    """
    EfficientNet specific initial block.

    Parameters:
    ----------
    nb_ch : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    activation : str
        Name of activation function.
    tf_mode : bool
        Whether to use TF-like mode.
    """

    def __init__(self,
                 nb_ch,
                 out_channels,
                 bn_eps,
                 activation,
                 tf_mode):
        super(EffiInitBlock, self).__init__()
        self.tf_mode = tf_mode

        self.conv = conv3x3_block(
            nb_ch=nb_ch,
            out_channels=out_channels,
            stride=1,
            padding=(0 if tf_mode else 1),
            bn_eps=bn_eps,
            activation=activation)

    def forward(self, x):
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=3, stride=2))
        x = self.conv(x)
        return x


class EfficientNet(nn.Module):
    """
    EfficientNet model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    kernel_sizes : list of list of int
        Number of kernel sizes for each unit.
    strides_per_stage : list int
        Stride value for the first unit of each stage.
    expansion_factors : list of list of int
        Number of expansion factors for each unit.
    dropout_rate : float, default 0.2
        Fraction of the input units to drop. Must be a number between 0 and 1.
    tf_mode : bool, default False
        Whether to use TF-like mode.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    nb_ch : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 kernel_sizes,
                 strides_per_stage,
                 expansion_factors,
                 dropout_rate=0.2,
                 tf_mode=False,
                 bn_eps=1e-5,
                 nb_ch=3,
                 in_size=(112, 112),
                 emb_size=512):
        super(EfficientNet, self).__init__()
        self.in_size = in_size
        activation = "prelu"

        self.features = nn.Sequential()
        self.features.add_module("init_block", EffiInitBlock(
            nb_ch=nb_ch,
            out_channels=init_block_channels,
            bn_eps=bn_eps,
            activation=activation,
            tf_mode=tf_mode))
        nb_ch = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            kernel_sizes_per_stage = kernel_sizes[i]
            expansion_factors_per_stage = expansion_factors[i]
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                kernel_size = kernel_sizes_per_stage[j]
                expansion_factor = expansion_factors_per_stage[j]
                stride = strides_per_stage[i] if (j == 0) else 1
                if i == 0:
                    stage.add_module("unit{}".format(j + 1), EffiDwsConvUnit(
                        nb_ch=nb_ch,
                        out_channels=out_channels,
                        stride=stride,
                        bn_eps=bn_eps,
                        activation=activation,
                        tf_mode=tf_mode))
                else:
                    stage.add_module("unit{}".format(j + 1), EffiInvResUnit(
                        nb_ch=nb_ch,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        exp_factor=expansion_factor,
                        se_factor=4,
                        bn_eps=bn_eps,
                        activation=activation,
                        tf_mode=tf_mode))
                nb_ch = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", conv1x1_block(
            nb_ch=nb_ch,
            out_channels=final_block_channels,
            bn_eps=bn_eps,
            activation=activation))
        nb_ch = final_block_channels
        #self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(output_size=1))

        '''
        self.output = nn.Sequential()
        if dropout_rate > 0.0:
            self.output.add_module("dropout", nn.Dropout(p=dropout_rate))
        self.output.add_module("fc", nn.Linear(
            in_features=nb_ch,
            out_features=num_classes))
        '''
        self.output = GDC(512, emb_size)
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def efficientnet(input_size, emb_size=512,nb_ch=3, version='b5', **kwargs):
    """
    Create EfficientNet model with specific parameters.
    """
    #assert input_size[0] in [112]
    if version.endswith('b') or version.endswith('c'):
        version = version[:-1]
        tf_mode = True
        bn_eps = 1e-3
    else:
        tf_mode = False
        bn_eps = 1e-5
    if version == "b0":
        in_size = (112, 112)
        depth_factor = 1.0
        width_factor = 1.0
        dropout_rate = 0.2
    elif version == "b1":
        in_size = (120, 120)
        depth_factor = 1.1
        width_factor = 1.0
        dropout_rate = 0.2
    elif version == "b2":
        in_size = (130, 130)
        depth_factor = 1.2
        width_factor = 1.1
        dropout_rate = 0.3
    elif version == "b3":
        in_size = (150, 150)
        depth_factor = 1.4
        width_factor = 1.2
        dropout_rate = 0.3
    elif version == "b4":
        in_size = (190, 190)
        depth_factor = 1.8
        width_factor = 1.4
        dropout_rate = 0.4
    elif version == "b5":
        in_size = (224, 224)
        depth_factor = 2.2
        width_factor = 1.6
        dropout_rate = 0.4
    elif version == "b6":
        in_size = (264, 264)
        depth_factor = 2.6
        width_factor = 1.8
        dropout_rate = 0.5
    elif version == "b7":
        in_size = (300, 300)
        depth_factor = 3.1
        width_factor = 2.0
        dropout_rate = 0.5
    elif version == "b8":
        in_size = (672, 672)
        depth_factor = 3.6
        width_factor = 2.2
        dropout_rate = 0.5
    else:
        raise ValueError("Unsupported EfficientNet version {}".format(version))

    init_block_channels = 32
    layers = [1, 2, 2, 3, 3, 4, 1]
    downsample = [1, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 40, 80, 112, 192, 320]
    expansion_factors_per_layers = [1, 6, 6, 6, 6, 6, 6]
    kernel_sizes_per_layers = [3, 3, 5, 3, 5, 5, 3]
    strides_per_stage = [1, 2, 2, 2, 1, 2, 1]
    final_block_channels = 512 # 1280

    layers = [int(math.ceil(li * depth_factor)) for li in layers]
    channels_per_layers = [round_channels(ci * width_factor) for ci in channels_per_layers]

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [])
    kernel_sizes = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                          zip(kernel_sizes_per_layers, layers, downsample), [])
    expansion_factors = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(expansion_factors_per_layers, layers, downsample), [])
    strides_per_stage = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(strides_per_stage, layers, downsample), [])
    strides_per_stage = [si[0] for si in strides_per_stage]

    init_block_channels = round_channels(init_block_channels * width_factor)

    #if width_factor > 1.0: #TODO: fix this!
    #    assert (int(final_block_channels * width_factor) == round_channels(final_block_channels * width_factor))
    #    final_block_channels = round_channels(final_block_channels * width_factor)

    net = EfficientNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernel_sizes=kernel_sizes,
        strides_per_stage=strides_per_stage,
        expansion_factors=expansion_factors,
        dropout_rate=dropout_rate,
        tf_mode=tf_mode,
        bn_eps=bn_eps,
        in_size=in_size,
        emb_size=emb_size,
        nb_ch=nb_ch,
        **kwargs)

    return net



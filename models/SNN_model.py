import torch
import torch.nn as nn
from train_net.layers import *
import torch.nn.functional as F
from torchvision import datasets, models
import torchsummary


class NMNISTNet(nn.Module):  # Example net for N-MNIST
    def __init__(self):
        super(NMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 20, 3, 1)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(7 * 7 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.conv1_s = tdLayer(self.conv1)
        self.pool1_s = tdLayer(self.pool1)
        self.conv2_s = tdLayer(self.conv2)
        self.pool2_s = tdLayer(self.pool2)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)

        self.spike = LIFSpike()

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = self.spike(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out


class MNISTNet(nn.Module):  # Example net for MNIST
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, 5, 1, 2, bias=None)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(15, 40, 5, 1, 2, bias=None)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(7 * 7 * 40, 300)
        self.fc2 = nn.Linear(300, 10)

        self.conv1_s = tdLayer(self.conv1)
        self.pool1_s = tdLayer(self.pool1)
        self.conv2_s = tdLayer(self.conv2)
        self.pool2_s = tdLayer(self.pool2)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)

        self.spike = LIFSpike()

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = self.spike(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out


class CifarNet(nn.Module):  # Example net for CIFAR10
    def __init__(self):
        super(CifarNet, self).__init__()
        # self.conv0 = nn.Conv2d(1, 1, 5, 2)
        self.conv0 = nn.Conv2d(3, 128, 3, 1, 1, bias=None)
        self.bn0 = tdBatchNorm(128)
        self.conv1 = nn.Conv2d(128, 256, 3, 1, 1, bias=None)
        self.bn1 = tdBatchNorm(256)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(256, 512, 3, 1, 1, bias=None)
        self.bn2 = tdBatchNorm(512)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(512, 1024, 3, 1, 1, bias=None)
        self.bn3 = tdBatchNorm(1024)
        self.conv4 = nn.Conv2d(1024, 512, 3, 1, 1, bias=None)
        self.bn4 = tdBatchNorm(512)
        self.fc1 = nn.Linear(8 * 8 * 512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        self.conv0_s = tdLayer(self.conv0, self.bn0)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.pool1_s = tdLayer(self.pool1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.pool2_s = tdLayer(self.pool2)
        self.conv3_s = tdLayer(self.conv3, self.bn3)
        self.conv4_s = tdLayer(self.conv4, self.bn4)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)
        self.fc3_s = tdLayer(self.fc3)

        self.spike = LIFSpike()

    def forward(self, x):
        x = self.conv0_s(x)
        x = self.spike(x)
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = self.spike(x)
        x = self.conv3_s(x)
        x = self.spike(x)
        x = self.conv4_s(x)
        x = self.spike(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        x = self.fc3_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / steps  # [N, neurons, steps]
        return out


# ------------------- #
#   ResNet Example    #
# ------------------- #


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, alpha=1 / (2 ** 0.5))
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.spike = LIFSpike()

    def forward(self, x):
        identity = x

        out = self.conv1_s(x)
        out = self.spike(out)
        out = self.conv2_s(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.spike(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))

        self.spike = LIFSpike()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, alpha=1 / (2 ** 0.5))
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1_s(x)
        x = self.spike(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        out = torch.sum(x, dim=4) / steps
        return out

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, progress, pretrained=None, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet19(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [3, 3, 2], pretrained, progress,
                   **kwargs)


class Decoder_4x_SNN(nn.Module):  # Example net for N-MNIST
    def __init__(self):
        super(Decoder_4x_SNN, self).__init__()
        self.up_block_s1 = up_block(1, 4)
        self.up_block_s2 = up_block(4, 10)
        self.conv1x1_s = tdLayer(nn.Conv2d(10, 1, 1, 1))

        self.spike = LIFSpike()

    def forward(self, x):
        x = self.up_block_s1(x)
        x = self.up_block_s2(x)
        x = self.conv1x1_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=4) / steps  # [N, neurons, steps]
        # out = F.softmax(out, dim=1)
        return out


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up_block, self).__init__()

        #  if your machine do not have enough memory to handle all those weights
        #  bilinear interpolation could be used to do the upsampling.
        if bilinear:
            self.up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up_layer = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)

        self.spike = LIFSpike()
        self.conv_s1 = tdLayer(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.conv_s2 = tdLayer(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.up_layer_s = tdLayer(self.up_layer)

    def forward(self, x):
        x = self.conv_s1(x)
        x = self.spike(x)
        x = self.conv_s2(x)
        x = self.spike(x)
        x = self.up_layer_s(x)
        x = self.spike(x)
        return x

class enconder_decoder_4x_SNN(nn.Module):
    def __init__(self):
        super(enconder_decoder_4x_SNN, self).__init__()
        self.down_s1 = down_layer(1, 8)
        self.down_s2 = down_layer(8, 32)
        self.down_s3 = down_layer(32, 128)
        self.up_s1 = up_layer(128, 64)
        self.up_s2 = up_layer(64, 32)
        self.up_s3 = up_layer(32, 16)
        self.up_s4 = up_layer(16, 8)
        self.up_s5 = up_layer(8, 4)
        self.conv1x1_s = tdLayer(nn.Conv2d(4, 1, 1, 1))
        self.spike = LIFSpike()

    def forward(self, x):
        x = self.down_s1(x)
        x = self.down_s2(x)
        x = self.down_s3(x)
        x = self.up_s1(x)
        x = self.up_s2(x)
        x = self.up_s3(x)
        x = self.up_s4(x)
        x = self.up_s5(x)
        x = self.conv1x1_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=4) / steps  # [N, neurons, steps]
        # out = F.softmax(out, dim=1)
        return out

class up_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up_layer, self).__init__()

        #  if your machine do not have enough memory to handle all those weights
        #  bilinear interpolation could be used to do the upsampling.
        if bilinear:
            self.up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up_layer = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.up_layer_s = tdLayer(self.up_layer, bn = tdBatchNorm(out_ch))
        self.spike = LIFSpike()

    def forward(self, x):
        x = self.up_layer_s(x)
        x = self.spike(x)
        return x

class down_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(down_layer, self).__init__()
        self.conv_s1 = tdLayer(nn.Conv2d(in_ch, out_ch, 3, padding=1),bn = tdBatchNorm(out_ch))
        self.spike = LIFSpike()
        self.average_pooling = tdLayer(nn.AvgPool2d(2,2))

    def forward(self, x):
        x = self.conv_s1(x)
        x = self.average_pooling(x)
        x = self.spike(x)
        return x


class Unet_8x_SNN(nn.Module):
    def __init__(self):
        super(Unet_8x_SNN, self).__init__()
        self.up_8x = tdLayer (nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))
        self.in_conv_s = tdLayer( nn.Conv2d(1, 16, 3, 1, 1),bn = tdBatchNorm(16) )
        self.down_s1 = down_layer(16, 32)
        self.down_s2 = down_layer(32, 64)
        self.down_s3 = down_layer(64, 128)
        self.up_s1 = Unet_up_layer(128, 64)
        self.up_s2 = Unet_up_layer(64, 32)
        self.up_s3 = Unet_up_layer(32, 16)
        self.out_conv_s = tdLayer(nn.Conv2d(16, 1, 1, 1))
        self.spike = LIFSpike()

    def forward(self, x):

        x = self.up_8x(x)
        x = self.in_conv_s(x)
        x = self.spike(x)
        x_down1 = self.down_s1(x)
        x_down2 = self.down_s2(x_down1)
        x_down3 = self.down_s3(x_down2)
        x_up1 = self.up_s1(x_down3, x_down2)
        x_up2 = self.up_s2(x_up1, x_down1)
        x = self.up_s3(x_up2, x)
        x = self.out_conv_s(x)
        # x = self.spike(x)
        out = torch.sum(x, dim=4) / steps  # [N, neurons, steps]
        # out = F.softmax(out, dim=1)
        return out

class Unet_up_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Unet_up_layer, self).__init__()

        #  if your machine do not have enough memory to handle all those weights
        #  bilinear interpolation could be used to do the upsampling.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.up_layer_s = tdLayer(self.up, bn = tdBatchNorm(out_ch))
        self.spike = LIFSpike()
        self.conv_s = tdLayer(nn.Conv2d(in_ch, out_ch, 3, 1, 1),bn = tdBatchNorm(out_ch))


    def forward(self, x1, x2):
        x1 = self.up_layer_s(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_s(x)
        self.spike(x)
        return x

# if __name__ == '__main__':
#     # net = Unet_4x_SNN()
#     # torchsummary.summary(net, input_size=[(3, 256, 256,1)], batch_size=5, device="cpu")
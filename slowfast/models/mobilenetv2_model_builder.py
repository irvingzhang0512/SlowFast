from torch import nn
from .build import MODEL_REGISTRY


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


def conv_3x3x3_bn_relu(inp, oup, stride, groups=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, (1, 1, 1), groups=1, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1x1_bn_relu(inp, oup, stride, groups=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, stride, 0, groups=1, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_3x1x1_bn_relu(inp, oup, stride, groups=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, (3, 1, 1), stride, (1, 0, 0), groups=1,
                  bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x3x3_bn_relu(inp, oup, stride, groups=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, (1, 3, 3), stride, (0, 1, 1), groups=1,
                  bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(conv_1x1x1_bn_relu(inp, hidden_dim, stride=1))
        layers.extend([
            # dw
            conv_3x3x3_bn_relu(hidden_dim,
                               hidden_dim,
                               stride=stride,
                               groups=hidden_dim),
            # pw-linear
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm3d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@MODEL_REGISTRY.register()
class MobileNetV2(nn.Module):
    def __init__(self, cfg):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()
        num_classes = cfg.MODEL.NUM_CLASSES
        round_nearest = cfg.MOBILENET.ROUND_NEAREST
        width_mult = cfg.MOBILENET.WIDTH_MULT
        block = InvertedResidual
        input_channel = cfg.MOBILENET.IN_CHANNELS
        last_channel = cfg.MOBILENET.OUT_CHANNELS

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (1, 2, 2)],
            [6, 32, 3, (1, 2, 2)],
            [6, 64, 4, (1, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (1, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(
                                 inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult,
                                        round_nearest)

        # STEM
        features = [conv_3x3x3_bn_relu(3, input_channel, stride=2)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                features.append(
                    block(input_channel,
                          output_channel,
                          stride,
                          expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest)
        features.append(
            conv_1x1x1_bn_relu(input_channel, self.last_channel, stride=1))

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x[0])
        x = x.mean([2, 3, 4])
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


if __name__ == '__main__':
    import torch
    model = MobileNetV2(400)
    print(model)
    res = model(torch.randn(16, 3, 16, 224, 224))
    print(res.shape, res)

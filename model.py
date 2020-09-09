import torch
import torch.nn as nn
import torch.nn.functional as F


# Reference
# https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, block_type='spatial', bias=False):
        super().__init__()
        if block_type == 'spatial':
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=bias)
        else:
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=bias)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=bias)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SlowFast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 8], n_classes=101, lateral_type='time_strided_conv', alpha=8, beta=8):
        super(SlowFast, self).__init__()
        self.alpha, self.beta = alpha, beta
        self.lateral_type = lateral_type

        self.slow_stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(True)
        )

        if lateral_type == 'time_strided_conv':
            self.slow_in_planes = 64 + 64 // self.beta * 2
            self.lateral_connection1 = nn.Conv3d(8, 8 * 2, kernel_size=(5, 1, 1), stride=(self.alpha, 1, 1), padding=(2, 0, 0))
            self.lateral_connection2 = nn.Conv3d(32, 32 * 2, kernel_size=(5, 1, 1), stride=(self.alpha, 1, 1), padding=(2, 0, 0))
            self.lateral_connection3 = nn.Conv3d(64, 64 * 2, kernel_size=(5, 1, 1), stride=(self.alpha, 1, 1), padding=(2, 0, 0))
            self.lateral_connection4 = nn.Conv3d(128, 128 * 2, kernel_size=(5, 1, 1), stride=(self.alpha, 1, 1), padding=(2, 0, 0))
        elif lateral_type == 'time2channel':
            self.slow_in_planes = 64 + 64 // self.beta * self.alpha
        elif lateral_type == 'time_strided_sampling':
            self.slow_in_planes = 64 + 64 // self.beta

        self.slow_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.slow_layer1 = self._make_layer(block, 64, layers[0], stride=1, block_type='spatial', path='slow')
        self.slow_layer2 = self._make_layer(block, 128, layers[1], stride=(1, 2, 2), block_type='spatial', path='slow')
        self.slow_layer3 = self._make_layer(block, 256, layers[2], stride=(1, 2, 2), block_type='temporal', path='slow')
        self.slow_layer4 = self._make_layer(block, 512, layers[3], stride=(1, 2, 2), block_type='temporal', path='slow')

        self.fast_in_planes = 8
        self.fast_stem = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(True)
        )
        self.fast_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_layer1 = self._make_layer(block, 8, layers[0], stride=1, block_type='temporal', path='fast')
        self.fast_layer2 = self._make_layer(block, 16, layers[1], stride=(1, 2, 2), block_type='temporal', path='fast')
        self.fast_layer3 = self._make_layer(block, 32, layers[2], stride=(1, 2, 2), block_type='temporal', path='fast')
        self.fast_layer4 = self._make_layer(block, 64, layers[3], stride=(1, 2, 2), block_type='temporal', path='fast')

        self.fast_gap = nn.AdaptiveAvgPool3d(1)
        self.slow_gap = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(2048 + self.fast_in_planes, n_classes)
        )

    def forward(self, x):
        slow_x, fast_x = self._data_layer(x)

        slow_out, fast_out = self.slow_stem(slow_x), self.fast_stem(fast_x)
        slow_out, fast_out = self.slow_pool(slow_out), self.fast_pool(fast_out)

        fast_out1 = self.fast_layer1(fast_out)
        fast_out2 = self.fast_layer2(fast_out1)
        fast_out3 = self.fast_layer3(fast_out2)
        fast_out4 = self.fast_layer4(fast_out3)

        if self.lateral_type == 'time_strided_conv':
            lateral_out1 = self.lateral_connection1(fast_out)
            lateral_out2 = self.lateral_connection2(fast_out1)
            lateral_out3 = self.lateral_connection3(fast_out2)
            lateral_out4 = self.lateral_connection4(fast_out3)
        elif self.lateral_type == 'time2channel':
            batch, C, temporal, width, height = fast_out.size()
            lateral_out1 = fast_out.view(batch, -1, temporal // self.alpha, width, height)
            batch, C, temporal, width, height = fast_out1.size()
            lateral_out2 = fast_out1.view(batch, -1, temporal // self.alpha, width, height)
            batch, C, temporal, width, height = fast_out2.size()
            lateral_out3 = fast_out2.view(batch, -1, temporal // self.alpha, width, height)
            batch, C, temporal, width, height = fast_out3.size()
            lateral_out4 = fast_out3.view(batch, -1, temporal // self.alpha, width, height)
        elif self.lateral_type == 'time_strided_sampling':
            lateral_out1 = fast_out[:, :, ::self.alpha, :, :]
            lateral_out2 = fast_out1[:, :, ::self.alpha, :, :]
            lateral_out3 = fast_out2[:, :, ::self.alpha, :, :]
            lateral_out4 = fast_out3[:, :, ::self.alpha, :, :]

        slow_out = self.slow_layer1(torch.cat([slow_out, lateral_out1], dim=1))
        slow_out = self.slow_layer2(torch.cat([slow_out, lateral_out2], dim=1))
        slow_out = self.slow_layer3(torch.cat([slow_out, lateral_out3], dim=1))
        slow_out = self.slow_layer4(torch.cat([slow_out, lateral_out4], dim=1))

        fast_out = self.fast_gap(fast_out4)
        slow_out = self.slow_gap(slow_out)

        out = torch.cat([slow_out, fast_out], dim=1).view(x.size(0), -1)
        out = self.fc(out)

        return out

    def _make_layer(self, block, planes, blocks, stride=(1, 1, 1), block_type='spatial', path='slow'):
        downsample = None
        layers = []

        if path == 'slow':
            if stride != 1 or self.slow_in_planes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv3d(self.slow_in_planes, planes * block.expansion, kernel_size=1, stride=stride),
                    nn.BatchNorm3d(planes * block.expansion))

            layers.append(
                block(in_planes=self.slow_in_planes,
                      planes=planes,
                      stride=stride,
                      downsample=downsample,
                      block_type=block_type))
            self.slow_in_planes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.slow_in_planes, planes))
            if self.lateral_type == 'time_strided_conv':
                self.slow_in_planes += 2 * planes * block.expansion // self.beta
            elif self.lateral_type == 'time2channel':
                self.slow_in_planes += planes * block.expansion // self.beta * self.alpha
            elif self.lateral_type == 'time_strided_sampling':
                self.slow_in_planes += planes * block.expansion // self.beta
        else:
            if stride != 1 or self.fast_in_planes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv3d(self.fast_in_planes, planes * block.expansion, kernel_size=1, stride=stride),
                    nn.BatchNorm3d(planes * block.expansion))

            layers.append(
                block(in_planes=self.fast_in_planes,
                      planes=planes,
                      stride=stride,
                      downsample=downsample,
                      block_type=block_type))
            self.fast_in_planes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.fast_in_planes, planes))

        return nn.Sequential(*layers)

    def _data_layer(self, x):
        # We opt to not perform temporal downsampling in this instantiation, as doing so would be detrimental when the input stride is large.
        slow_data = x[:, :, ::16, :, :]
        fast_data = x[:, :, ::2, :, :]

        return slow_data, fast_data


def ResNet50(block=Bottleneck, layers=[3, 4, 6, 8], n_classes=101):
    return SlowFast(block=block, layers=layers, n_classes=n_classes)


def main():
    sf = SlowFast()
    x = torch.randn([2, 3, 64, 224, 224])

    sf(x)


if __name__ == '__main__':
    main()

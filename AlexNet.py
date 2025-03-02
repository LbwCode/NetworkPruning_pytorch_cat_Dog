from torch import nn
import torch
from torch.autograd import Variable


class AlexNet(nn.Module):
    def __init__(self, cfg=None):
        super(AlexNet, self).__init__()
        # cfg =

        if cfg is None:
            cfg = [64, 'Max', 192, 'Max', 384, 256, 256, 'Max']
        self.make_ = self.make_def(cfg)

        self.f1 = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.f2 = nn.Dropout(p=0.5, inplace=False)
        self.f3 = nn.Flatten()

        self.f4 = nn.Linear(cfg[-2]*6*6, out_features=4096, bias=True)
        self.f5 = nn.ReLU(inplace=True)
        self.f6 = nn.Dropout(p=0.5, inplace=False)
        self.f7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.f8 = nn.ReLU(inplace=True)
        self.f9 = nn.Linear(in_features=4096, out_features=10, bias=True)

    def make_def(self, cfg):
        n = 3
        layers = []
        for i, v in enumerate(cfg):
            if v == "Max":
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)]
            elif i == 0:
                conv2d = nn.Conv2d(n, v, kernel_size=(11, 11), stride=(1, 1), padding=(1, 1))
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                n = v
            elif i == 2:
                conv2d = nn.Conv2d(n, v, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                n = v
            else:
                conv2d = nn.Conv2d(n, v, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                n = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.make_(x)
        x = self.f1(x)
        x = self.f2(x)
        # print(x.shape)
        x = self.f3(x)

        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        x = self.f9(x)
        return x


if __name__ == '__main__':
    net = AlexNet()
    x = Variable(torch.FloatTensor(16, 3, 32, 32))
    y = net(x)
    print(y.data.shape)

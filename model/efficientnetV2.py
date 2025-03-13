import torch
from timm.models.layers import DropPath
from torch.nn.functional import cross_entropy, dropout, one_hot, softmax


def init_weight(model):
    import math
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out // m.groups
            torch.nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, torch.nn.Linear):
            init_range = 1.0 / math.sqrt(m.weight.size()[0])
            torch.nn.init.uniform_(m.weight, -init_range, init_range)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, (k - 1) // 2, 1, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class SE(torch.nn.Module):
    def __init__(self, ch, r):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Conv2d(ch, ch // (4 * r), 1),
                                      torch.nn.SiLU(),
                                      torch.nn.Conv2d(ch // (4 * r), ch, 1),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


class Residual(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1801.04381.pdf]
    """

    def __init__(self, in_ch, out_ch, s, r, dp_rate=0, fused=True):
        super().__init__()
        identity = torch.nn.Identity()
        self.add = s == 1 and in_ch == out_ch

        if fused:
            features = [Conv(in_ch, r * in_ch, activation=torch.nn.SiLU(), k=3, s=s),
                        Conv(r * in_ch, out_ch, identity) if r != 1 else identity,
                        DropPath(dp_rate) if self.add else identity]
        else:
            features = [Conv(in_ch, r * in_ch, torch.nn.SiLU()) if r != 1 else identity,
                        Conv(r * in_ch, r * in_ch, torch.nn.SiLU(), 3, s, r * in_ch),
                        SE(r * in_ch, r), Conv(r * in_ch, out_ch, identity),
                        DropPath(dp_rate) if self.add else identity]

        self.res = torch.nn.Sequential(*features)

    def forward(self, x):
        return x + self.res(x) if self.add else self.res(x)


class EfficientNet(torch.nn.Module):
    """
     efficientnet-v2-s :
                        num_dep = [2, 4, 4, 6, 9, 15, 0]
                        filters = [24, 48, 64, 128, 160, 256, 256, 1280]
     efficientnet-v2-m :
                        num_dep = [3, 5, 5, 7, 14, 18, 5]
                        filters = [24, 48, 80, 160, 176, 304, 512, 1280]
     efficientnet-v2-l :
                        num_dep = [4, 7, 7, 10, 19, 25, 7]
                        filters = [32, 64, 96, 192, 224, 384, 640, 1280]
    """

    def __init__(self, drop_rate=0, num_class=1000):
        super().__init__()
        num_dep = [2, 4, 4, 6, 9, 15, 0]
        filters = [24, 48, 64, 128, 160, 256, 256, 1280]

        dp_index = 0
        dp_rates = [x.item() for x in torch.linspace(0, 0.2, sum(num_dep))]

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        for i in range(num_dep[0]):
            if i == 0:
                self.p1.append(Conv(3, filters[0], torch.nn.SiLU(), 3, 2))
                self.p1.append(Residual(filters[0], filters[0], 1, 1, dp_rates[dp_index]))
            else:
                self.p1.append(Residual(filters[0], filters[0], 1, 1, dp_rates[dp_index]))
            dp_index += 1
        # p2/4
        for i in range(num_dep[1]):
            if i == 0:
                self.p2.append(Residual(filters[0], filters[1], 2, 4, dp_rates[dp_index]))
            else:
                self.p2.append(Residual(filters[1], filters[1], 1, 4, dp_rates[dp_index]))
            dp_index += 1
        # p3/8
        for i in range(num_dep[2]):
            if i == 0:
                self.p3.append(Residual(filters[1], filters[2], 2, 4, dp_rates[dp_index]))
            else:
                self.p3.append(Residual(filters[2], filters[2], 1, 4, dp_rates[dp_index]))
            dp_index += 1
        # p4/16
        for i in range(num_dep[3]):
            if i == 0:
                self.p4.append(Residual(filters[2], filters[3], 2, 4, dp_rates[dp_index], False))
            else:
                self.p4.append(Residual(filters[3], filters[3], 1, 4, dp_rates[dp_index], False))
            dp_index += 1
        for i in range(num_dep[4]):
            if i == 0:
                self.p4.append(Residual(filters[3], filters[4], 1, 6, dp_rates[dp_index], False))
            else:
                self.p4.append(Residual(filters[4], filters[4], 1, 6, dp_rates[dp_index], False))
            dp_index += 1
        # p5/32
        for i in range(num_dep[5]):
            if i == 0:
                self.p5.append(Residual(filters[4], filters[5], 2, 6, dp_rates[dp_index], False))
            else:
                self.p5.append(Residual(filters[5], filters[5], 1, 6, dp_rates[dp_index], False))
            dp_index += 1
        for i in range(num_dep[6]):
            if i == 0:
                self.p5.append(Residual(filters[5], filters[6], 2, 6, dp_rates[dp_index], False))
            else:
                self.p5.append(Residual(filters[6], filters[6], 1, 6, dp_rates[dp_index], False))
            dp_index += 1

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

        self.fc1 = torch.nn.Sequential(Conv(filters[6], filters[7], torch.nn.SiLU()),
                                       torch.nn.AdaptiveAvgPool2d(1),
                                       torch.nn.Flatten())
        self.fc2 = torch.nn.Linear(filters[7], num_class)

        self.drop_rate = drop_rate

        init_weight(self)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)

        x = self.fc1(x)
        if self.drop_rate > 0:
            x = dropout(x, self.drop_rate, self.training)
        return self.fc2(x)

    def export(self):
        from timm.models.layers import Swish
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'relu'):
                if isinstance(m.relu, torch.nn.SiLU):
                    m.relu = Swish()
            if type(m) is SE:
                if isinstance(m.se[2], torch.nn.SiLU):
                    m.se[2] = Swish()
        return self


class EMA:
    def __init__(self, model, decay=0.9999):
        super().__init__()
        import copy
        self.decay = decay
        self.model = copy.deepcopy(model)

        self.model.eval()

    def update_fn(self, model, fn):
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.module.state_dict().values()
            for e, m in zip(e_std, m_std):
                e.copy_(fn(e, m))

    def update(self, model):
        self.update_fn(model, fn=lambda e, m: self.decay * e + (1. - self.decay) * m)


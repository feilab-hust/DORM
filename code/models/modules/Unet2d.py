import torch.nn as nn
import torch
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, nom_='instance'):
        super(DoubleConv, self).__init__()
        if nom_ == 'batch':
            self.NormLayer = nn.BatchNorm2d(out_ch)
        elif nom_ == 'instance':
            self.NormLayer = nn.InstanceNorm2d(out_ch)
        else:
            raise NotImplementedError('{} not implemented'.format(nom_))

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            self.NormLayer,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            self.NormLayer,
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, uptype='upsample'):
        super(UpConv, self).__init__()
        if uptype == 'transpose':
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            )
        elif uptype == 'upsample':
            self.conv = nn.Sequential(
                UpsampleBlock(in_ch, out_ch)
            )
        else:
            raise NotImplementedError('{} not implemented'.format(uptype))

    def forward(self, input):
        return self.conv(input)


class HeadConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HeadConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=0))

    def forward(self, x):
        x = torch.nn.functional.pad(x, (2, 2, 2, 2), mode='reflect')
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(
            self, n_feats=1, kernel_size=3, act=(nn.ReLU(True)), res_scale=1, nom_='instance'):
        super(ResBlock, self).__init__()
        if nom_ == 'batch':
            self.NormLayer = nn.BatchNorm2d(n_feats)
        elif nom_ == 'instance':
            self.NormLayer = nn.InstanceNorm2d(n_feats)
        else:
            raise NotImplementedError('{} not implemented'.format(nom_))
        self.body1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=0),
            self.NormLayer,
            act
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=0),
            self.NormLayer)
        self.res_scale = res_scale

    def forward(self, x):
        x0 = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
        x1 = self.body1(x0)
        x2 = torch.nn.functional.pad(x1, (1, 1, 1, 1), mode='reflect')
        res = self.body2(x2).mul(self.res_scale)
        res += x
        return res


class Bodyconv(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, num_b=6, nom_='instance'):
        super(Bodyconv, self).__init__()
        self.num_b = num_b
        self.f_con = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.layers = nn.ModuleList([ResBlock(in_ch, nom_=nom_) for _ in range(self.num_b)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x_out = self.f_con(x)
        return x_out


class Unet2d(nn.Module):
    def __init__(self, num_features=32, in_ch=1, out_ch=1, uptype='subpixel', nom_='instance'):
        super(Unet2d, self).__init__()

        # self.conv1 = DoubleConv(in_ch, num_features)
        self.conv1 = HeadConv(in_ch, num_features)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(num_features, 2 * num_features, nom_=nom_)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = Bodyconv(2 * num_features, 4 * num_features, nom_=nom_)

        self.up8 = UpConv(4 * num_features, 2 * num_features, uptype)
        self.conv8 = DoubleConv(4 * num_features, 2 * num_features, nom_=nom_)
        self.up9 = UpConv(2 * num_features, 1 * num_features, uptype)
        self.conv9 = DoubleConv(2 * num_features, 1 * num_features, nom_=nom_)
        self.conv10 = nn.Conv2d(num_features, out_ch, 1)

    def forward(self, x):
        x_raw = x
        # print(x.shape)
        # x = torch.nn.functional.pad(x, (2, 2, 2, 2, 2, 2), mode='reflect')
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # print(p2.shape)
        c3 = self.conv3(p2)

        up_8 = self.up8(c3)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # out = nn.LeakyReLU()(c10)
        out = nn.LeakyReLU(0.2, True)(c10)
        return out


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    # net = Unet2d(num_features=32, uptype='subpixel')
    # net = Unet2d(num_features=32, uptype='transpose')
    net = Unet2d(num_features=16, uptype='upsample')
    model_info(net)

    x = torch.randn(2, 1, 32, 32)
    x = net(x)
    print(x.shape)

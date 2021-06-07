import torch
import torch.nn as nn
from torch.autograd import Variable
import functools


def gaussian(tensor, mean=0, stddev=0.01):
    return Variable(tensor + torch.randn(tensor.size()) * stddev + mean)


class HideNet(nn.Module):
    def __init__(self):
        super(HideNet, self).__init__()

        unet_block = HideNetSkipConnectionBlock(64 * 8, 64 * 8, innermost=True)
        for i in range(7 - 5):
            unet_block = HideNetSkipConnectionBlock(64 * 8, 64 * 8, submodule=unet_block)

        unet_block = HideNetSkipConnectionBlock(64 * 4, 64 * 8, submodule=unet_block)
        unet_block = HideNetSkipConnectionBlock(64 * 2, 64 * 4, submodule=unet_block)
        unet_block = HideNetSkipConnectionBlock(64, 64 * 2,  submodule=unet_block)
        unet_block = HideNetSkipConnectionBlock(3, 64, input_nc=6, submodule=unet_block, outermost=True)
        self.model = unet_block

    def forward(self, secret_concat_cover):
        return self.model(secret_concat_cover)


class HideNetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d):
        super(HideNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class RevealNet(nn.Module):
    def __init__(self):
        super(RevealNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64 * 2, 3, 1, 1),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
            nn.Conv2d(64 * 2, 64 * 4, 3, 1, 1),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            nn.Conv2d(64 * 4, 64 * 2, 3, 1, 1),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
            nn.Conv2d(64 * 2, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, c_prime):
        revealed = self.main(c_prime)
        return revealed

import numpy as np
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.loss import VGG19
from src.network.networks import Discriminator, avgcov2d_layer, conv2d_layer
from torch.autograd import Variable


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='identity', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(
                        m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "identity":
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, gain)
                    else:
                        identity_initializer(m.weight.data)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


def identity_initializer(data):
    shape = data.shape
    array = np.zeros(shape, dtype=float)
    cx, cy = shape[2]//2, shape[3]//2
    for i in range(np.minimum(shape[0], shape[1])):
        array[i, i, cx, cy] = 1
    return torch.tensor(array, dtype=torch.float32)


class TBRNet(BaseNetwork):
    """TBRNet"""

    def __init__(self, config, in_channels=3, init_weights=True):
        super(TBRNet, self).__init__()

        # gan
        self.network = TBRNetSOURCE(
            in_channels, 64, norm=config.GAN_NORM, stage_num=[6,2])
        # gan loss
        if config.LOSS == "MSELoss":
            self.add_module('loss', nn.MSELoss(reduction="mean"))
        elif config.LOSS == "L1Loss":
            self.add_module('loss', nn.L1Loss(reduction="mean"))

        # gan optimizer
        self.optimizer = optim.Adam(
            params=self.network.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        # dis
        self.ADV = config.ADV
        if self.ADV:
            discriminator = []
            discriminator.append(Discriminator(config, in_channels=6))
            self.discriminator = nn.Sequential(*discriminator)

        if init_weights:
            self.init_weights(config.INIT_TYPE)

    def process(self, images, mask, GT):
        loss = []
        logs = []
        inputs = images
        img, net_matte, outputs = self(inputs)

        # cal_loss
        match_loss_1 = self.cal_loss(outputs, GT)*255*10
        match_loss_2 = self.cal_loss(img, images)*255*10

        matte_gt = GT-inputs
        matte_gt = matte_gt - \
            (matte_gt.min(dim=2, keepdim=True).values).min(
                dim=3, keepdim=True).values
        matte_gt = matte_gt / \
            (matte_gt.max(dim=2, keepdim=True).values).max(
                dim=3, keepdim=True).values
        matte_loss = self.cal_loss(net_matte, matte_gt)*255

        if self.ADV:
            dis_loss_1, gen_gan_loss_1, perceptual_loss_1 = self.discriminator[0].cal_loss(
                images, outputs, GT)

            perceptual_loss_1 = perceptual_loss_1*1000

            gen_loss = perceptual_loss_1 +match_loss_1+match_loss_2+\
                matte_loss+gen_gan_loss_1

            loss.append(gen_loss)
            loss.append(dis_loss_1)

            logs.append(("l_match1", match_loss_1.item()))
            logs.append(("l_match2", match_loss_2.item()))
            logs.append(("l_matte", matte_loss.item()))
            logs.append(("l_perceptual_1", perceptual_loss_1.item()))
            logs.append(("l_adv1", gen_gan_loss_1.item()))
            logs.append(("l_gen", gen_loss.item()))
            logs.append(("l_dis1", dis_loss_1.item()))
        else:
            gen_loss = match_loss_1 + match_loss_2 + matte_loss
            gen_loss = gen_loss
            loss.append(gen_loss)

            logs.append(("l_match1", match_loss_1.item()))
            logs.append(("l_match2", match_loss_2.item()))
            logs.append(("l_matte", matte_loss.item()))
            logs.append(("l_gen", gen_loss.item()))

        return [net_matte, outputs], loss, logs

    def forward(self, x):
        outputs = self.network(x)
        return outputs

    def cal_loss(self, outputs, GT):
        matching_loss = self.loss(outputs, GT)
        return matching_loss

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss[0].backward()
        self.optimizer.step()
        if self.ADV:
            i = 0
            for discriminator in self.discriminator:
                discriminator.backward(loss[1+i])
                i += 1


class TBRNetSOURCE(nn.Module):
    def __init__(self, in_channels=3, channels=64, norm="batch", stage_num=[6,2]):
        super(TBRNetSOURCE, self).__init__()
        self.stage_num = stage_num

        # Pre-trained VGG19
        self.add_module('vgg19', VGG19())

        # SE
        cat_channels = in_channels+64+128+256+512+512
        self.se = SELayer(cat_channels)
        self.conv_up = conv2d_layer(
            cat_channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        self.conv_mid = conv2d_layer(
            cat_channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        self.conv_down = conv2d_layer(
            cat_channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)

        # coarse
        self.down_sample = conv2d_layer(
            channels, 2*channels, kernel_size=4, stride=2, padding=1, dilation=1, norm=norm)
        coarse_list = []
        for i in range(self.stage_num[0]):
            coarse_list.append(TBR(2*channels, norm, mid_dilation=2**(i % 6)))
        self.coarse_list = nn.Sequential(*coarse_list)

        self.up_conv = conv2d_layer(
            2*channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, norm=norm)

        # refine
        fine_list = []
        for i in range(self.stage_num[1]):
            fine_list.append(TBR(channels, norm, mid_dilation=2**(i % 6)))
        self.fine_list = nn.Sequential(*fine_list)

        self.se_coarse = nn.Sequential(SELayer(2*channels),
                                       conv2d_layer(2*channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm))

        self.spp1 = SPP(channels, norm=norm)
        self.spp2 = SPP(channels, norm=norm)
        self.spp3 = SPP(channels, norm=norm)

        self.toimg1 = conv2d_layer(
            channels, 3, kernel_size=1,  padding=0, dilation=1, norm="none", activation_fn="Sigmoid")
        self.toimg2 = conv2d_layer(
            channels, 3, kernel_size=1,  padding=0, dilation=1, norm="none", activation_fn="Sigmoid")
        self.toimg3 = conv2d_layer(
            channels, 3, kernel_size=1,  padding=0, dilation=1, norm="none", activation_fn="Sigmoid")
    def forward(self, x):
        size = (x.shape[2], x.shape[3])

        # vgg
        x_vgg = self.vgg19(x)

        # hyper-column features
        x_cat = torch.cat((
            x,
            F.interpolate(x_vgg['relu1_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu2_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu3_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu4_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu5_2'], size, mode="bilinear", align_corners=True)), dim=1)

        # SE
        x_se = self.se(x_cat)
        x_up = self.conv_up(x_se)
        x_mid = self.conv_mid(x_se)
        x_down = self.conv_down(x_se)
        x_up_ = x_up
        x_mid_ = x_mid
        x_down_ = x_down
 
        # process
        x_up = self.down_sample(x_up)
        x_mid = self.down_sample(x_mid)
        x_down = self.down_sample(x_down)
        for i in range(self.stage_num[0]):
            x_up, x_mid, x_down = self.coarse_list[i](x_up, x_mid, x_down)

        x_up = F.interpolate(x_up, size, mode="bilinear", align_corners=True)
        x_up = self.up_conv(x_up)
        x_mid = F.interpolate(x_mid, size, mode="bilinear", align_corners=True)
        x_mid = self.up_conv(x_mid)
        x_down = F.interpolate(x_down, size, mode="bilinear", align_corners=True)
        x_down = self.up_conv(x_down)

        x_up = self.se_coarse(torch.cat((x_up_, x_up), dim=1))
        x_mid = self.se_coarse(torch.cat((x_mid_, x_mid), dim=1))
        x_down = self.se_coarse(torch.cat((x_down_, x_down), dim=1))
        for i in range(self.stage_num[1]):
            x_up, x_mid, x_down = self.fine_list[i](x_up, x_mid, x_down)

        # spp
        img = self.spp1(x_up)
        matte_out = self.spp2(x_mid)
        img_free = self.spp3(x_down)

        # output
        img = self.toimg1(img)
        matte_out = self.toimg2(matte_out)
        img_free = self.toimg3(img_free)

        return [img, matte_out, img_free]


class TBR(nn.Module):
    def __init__(self, channels=64, norm="batch", mid_dilation=1):
        super(TBR, self).__init__()
        # up
        self.conv_up = conv2d_layer(
            channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        avg_pool_down_mid = []
        avg_pool_down_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv2d_layer(channels, channels, kernel_size=1,  padding=0, dilation=1, norm="none")))
        avg_pool_down_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            conv2d_layer(channels, channels, kernel_size=3,  padding=0, dilation=1, norm="none")))
        avg_pool_down_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(5),
            conv2d_layer(channels, channels, kernel_size=5,  padding=0, dilation=1, norm="none")))
        avg_pool_down_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            conv2d_layer(channels, channels, kernel_size=7,  padding=0, dilation=1, norm="none")))
        self.avg_pool_down_mid = nn.Sequential(*avg_pool_down_mid)

        # mid
        self.conv_mid = conv2d_layer(
            channels, channels, kernel_size=3,  padding=mid_dilation, dilation=mid_dilation, norm=norm)

        # down
        self.conv_down = conv2d_layer(
            channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        avg_pool_up_mid = []
        avg_pool_up_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv2d_layer(channels, channels, kernel_size=1,  padding=0, dilation=1, norm="none")))
        avg_pool_up_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            conv2d_layer(channels, channels, kernel_size=3,  padding=0, dilation=1, norm="none")))
        avg_pool_up_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(5),
            conv2d_layer(channels, channels, kernel_size=5,  padding=0, dilation=1, norm="none")))
        avg_pool_up_mid.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            conv2d_layer(channels, channels, kernel_size=7,  padding=0, dilation=1, norm="none")))
        self.avg_pool_up_mid = nn.Sequential(*avg_pool_up_mid)

        # conv
        self.conv_up_mid = conv2d_layer(
            channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        self.conv_up_mid_1x1 = conv2d_layer(
            2*channels, channels, kernel_size=1,  padding=0, dilation=1, norm=norm)
        self.conv_down_mid = conv2d_layer(
            channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm)
        self.conv_down_mid_1x1 = conv2d_layer(
            2*channels, channels, kernel_size=1,  padding=0, dilation=1, norm=norm)

    def forward(self, up_in, mid_in, down_in):
        # up
        x_up = self.conv_up(up_in)
        x_down_mid = self.conv_down_mid(down_in-mid_in)
        pooling_up = 0
        for i in range(len(self.avg_pool_down_mid)):
            pooling_up = pooling_up + self.avg_pool_down_mid[i](x_down_mid)
        x_up = pooling_up*x_up + x_up + x_down_mid*x_up + up_in

        # mid
        x_mid = self.conv_mid(mid_in)
        x_mid = x_mid + mid_in

        # down
        x_down = self.conv_down(down_in)
        x_up_mid = self.conv_up_mid(up_in+mid_in)
        pooling_down = 0
        for i in range(len(self.avg_pool_up_mid)):
            pooling_down = pooling_down + self.avg_pool_up_mid[i](x_up_mid)
        x_down = pooling_down*x_down + x_down + x_up_mid*x_down + down_in

        return x_up, x_mid, x_down


class SPP(nn.Module):
    # SPP SOURCE - tensorflow http://github.com/vinthony/ghost-free-shadow-removal/
    def __init__(self, channels=64, norm="batch"):
        super(SPP, self).__init__()
        self.net2 = avgcov2d_layer(
            4, 4, channels, channels, 1, padding=0, norm=norm)
        self.net8 = avgcov2d_layer(
            8, 8, channels, channels, 1, padding=0, norm=norm)
        self.net16 = avgcov2d_layer(
            16, 16, channels, channels, 1, padding=0, norm=norm)
        self.net32 = avgcov2d_layer(
            32, 32, channels, channels, 1, padding=0, norm=norm)
        self.output = conv2d_layer(channels*5, channels, 3, norm=norm)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = torch.cat((
            F.interpolate(self.net2(x), size, mode="bilinear",
                          align_corners=True),
            F.interpolate(self.net8(x), size, mode="bilinear",
                          align_corners=True),
            F.interpolate(self.net16(x), size,
                          mode="bilinear", align_corners=True),
            F.interpolate(self.net32(x), size,
                          mode="bilinear", align_corners=True),
            x), dim=1)
        x = self.output(x)
        return x


class SELayer(nn.Module):
    # https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py -> https://github.com/vinthony/ghost-free-shadow-removal/blob/master/networks.py
    # reduction=16 -> reduction=8
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

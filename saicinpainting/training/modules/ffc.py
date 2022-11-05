# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from saicinpainting.training.modules.base import get_activation, BaseDiscriminator
from saicinpainting.training.modules.spatial_transform import LearnableSpatialTransformWrapper
from saicinpainting.training.modules.squeeze_excitation import SELayer
from saicinpainting.utils import get_shape


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return
        # print(torch.cat(x, dim=1).shape)
        # exit(0)
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        # # print(input.shape)
        # remove_number = 0
        # remove_start, remove_end = 13-remove_number, 14
        # for number, single_model in enumerate(self.model):
        #     if number>remove_start and  number<remove_end:
        #         continue
        #     # print(number, single_model)
        #     input = single_model(input)
            
        #     # if isinstance(input, torch.Tensor):
        #     #     print(torch.tensor(input).shape)
        #     # else: # type is tuple
        #     #     if isinstance(input[0], torch.Tensor):
        #     #         print(input[0].shape, end=' ')
        #     #     if isinstance(input[1], torch.Tensor):
        #     #         print(input[1].shape, end=' ')
        #     #     print('')
        #     #     # print(type(input[0]), type(input[1]))
        #     #     # [print(len(a)) for a in input]
        # return input
        return self.model(input)

class TsaDownFFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(TsaDownFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l_alpha = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g_alpha = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        # module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        # self.convg2g_alpha = module(
        #     in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        # self.global_alpha = nn.Parameter(torch.ones(out_cg, in_cg, 1, 1))
        # self.global_alpha.requires_grad = True

        

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            # tas_convl2l =  + F.conv2d(x_l, self.convl2l_alpha, stride=self.convl2l.stride)
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate + self.convl2l_alpha(x_l)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g) + self.convl2g_alpha(x_l) * l2g_gate

        return out_xl, out_xg

class TsaDownFFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(TsaDownFFC_BN_ACT, self).__init__()
        self.ffc = TsaDownFFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class TsaG2GDownUpFFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 TsaDownFFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [TsaDownFFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TsaG2GFFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]
        model_up_alpha = []
        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]
            model_up_alpha += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1)]

        if out_ffc:
            model += [TsaG2GFFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_up_alpha += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)
        self.model_up_alpha = nn.Sequential(*model_up_alpha)

    def forward(self, input, pa=False):
        for block in self.model[:-12]:
            input = block(input)
        # block_class_name = self.model[block].__class__.__name__
        #         if block_class_name == 'FFC_FSMR_BN_ACT' or block_class_name == 'FFC_FSMR_ResnetBlock':
        #             input = self.model[block](input, use_fsmr=True, shuffle_indices=shuffle_indices, alpha=alpha, only_x_g=only_x_g, only_some_imgs=only_some_imgs)
        #         elif block_class_name == 'ConvTranspose2d' or block_class_name == 'Conv2d':
        #             input = self.model[block](input)
        #             y_input =  input[shuffle_indices]
        #             input = FSM(input, y_input, alpha)
        up_loc = 0
        for block in self.model[-12:]:
            if block.__class__.__name__ == 'ConvTranspose2d' or block.__class__.__name__ == 'Conv2d':
                input = block(input) + self.model_up_alpha[up_loc](input)
                up_loc += 1
            else:
                input = block(input)
        return input

class TsaMiddleAllFFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(TsaMiddleAllFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l_alpha = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g_alpha = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l_alpha = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g_alpha = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            # tas_convl2l =  + F.conv2d(x_l, self.convl2l_alpha, stride=self.convl2l.stride)
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate + self.convl2l_alpha(x_l) + self.convg2l_alpha(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g) + self.convg2g_alpha(x_g) + self.convl2g_alpha(x_l) * l2g_gate

        return out_xl, out_xg


class TsaMiddleAllFFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(TsaMiddleAllFFC_BN_ACT, self).__init__()
        self.ffc = TsaMiddleAllFFC(in_channels, out_channels, kernel_size,
                                    ratio_gin, ratio_gout, stride, padding, dilation,
                                    groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class TsaMiddleAllConvFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = TsaMiddleAllFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                             norm_layer=norm_layer,
                                             activation_layer=activation_layer,
                                             padding_type=padding_type,
                                             **conv_kwargs)
        self.conv2 = TsaMiddleAllFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                             norm_layer=norm_layer,
                                             activation_layer=activation_layer,
                                             padding_type=padding_type,
                                             **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class TsaMiddleAllConvFFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TsaMiddleAllConvFFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                                       activation_layer=activation_layer,
                                                       norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [TsaMiddleAllConvFFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                                  norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

        # self.pa = pa(feat_dim=4)

    def forward(self, input, pa=False):
        # if pa:
        #     input = self.pa(input)
        return self.model(input)


class TsaG2GConvL2GFFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(TsaG2GConvL2GFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g_alpha = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g_alpha = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            # tas_convl2l =  + F.conv2d(x_l, self.convl2l_alpha, stride=self.convl2l.stride)
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g) + self.convg2g_alpha(x_g) + self.convl2g_alpha(x_l) * l2g_gate

        return out_xl, out_xg


class TsaG2GConvL2GFFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(TsaG2GConvL2GFFC_BN_ACT, self).__init__()
        self.ffc = TsaG2GConvL2GFFC(in_channels, out_channels, kernel_size,
                                    ratio_gin, ratio_gout, stride, padding, dilation,
                                    groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class TsaG2GConvL2GFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = TsaG2GConvL2GFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                             norm_layer=norm_layer,
                                             activation_layer=activation_layer,
                                             padding_type=padding_type,
                                             **conv_kwargs)
        self.conv2 = TsaG2GConvL2GFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                             norm_layer=norm_layer,
                                             activation_layer=activation_layer,
                                             padding_type=padding_type,
                                             **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class TsaG2GConvL2GFFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TsaG2GConvL2GFFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                                       activation_layer=activation_layer,
                                                       norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [TsaG2GConvL2GFFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                                  norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

        # self.pa = pa(feat_dim=4)

    def forward(self, input, pa=False):
        # if pa:
        #     input = self.pa(input)
        return self.model(input)


class TsaG2GConvL2LFFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(TsaG2GConvL2LFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l_alpha = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g_alpha = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)
        

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            # tas_convl2l =  + F.conv2d(x_l, self.convl2l_alpha, stride=self.convl2l.stride)
            out_xl = self.convl2l(x_l) + self.convl2l_alpha(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g) + self.convg2g_alpha(x_g)        

        return out_xl, out_xg

class TsaG2GConvL2LFFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(TsaG2GConvL2LFFC_BN_ACT, self).__init__()
        self.ffc = TsaG2GConvL2LFFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class TsaG2GConvL2LFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = TsaG2GConvL2LFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = TsaG2GConvL2LFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out

class TsaG2GConvL2LFFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TsaG2GConvL2LFFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [TsaG2GConvL2LFFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

        # self.pa = pa(feat_dim=4)

    def forward(self, input, pa=False):
        # if pa:
        #     input = self.pa(input)
        return self.model(input)

class TsaG2GL2LFFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(TsaG2GL2LFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g_alpha = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.convl2l_alpha = nn.Parameter(torch.ones(out_cl, in_cl, 1, 1))
        self.convl2l_alpha.requires_grad = True
        

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            tas_convl2l = self.convl2l(x_l) + F.conv2d(x_l, self.convl2l_alpha, stride=self.convl2l.stride)
            out_xl = tas_convl2l + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g) + self.convg2g_alpha(x_g)        

        return out_xl, out_xg

class TsaG2GL2GFFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(TsaG2GL2GFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g_alpha = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.convl2g_alpha = nn.Parameter(torch.ones(out_cg, in_cl, 1, 1))
        self.convl2g_alpha.requires_grad = True
        

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            tsa_convl2g = self.convl2g(x_l) + F.conv2d(x_l, self.convl2g_alpha, stride=self.convl2g.stride)
            out_xg = tsa_convl2g * l2g_gate + self.convg2g(x_g) + self.convg2g_alpha(x_g)        

        return out_xl, out_xg

class TsaG2GFFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(TsaG2GFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g_alpha = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)
        

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g) + self.convg2g_alpha(x_g)        

        return out_xl, out_xg

class TsaOneFFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(TsaOneFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

        # self.sepc_trans_alpha = nn.Parameter(torch.ones(out_cg, in_cg, 1, 1))
        # self.sepc_trans_alpha.requires_grad = True

        self.global_alpha = nn.Parameter(torch.ones(out_cg, in_cg, 1, 1))
        self.global_alpha.requires_grad = True
        

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
            out_xg = out_xg + F.conv2d(x_g, self.global_alpha, stride=self.convg2g.stride)
            # out_g2g = self.convg2g(x_g)
            # out_xg = self.convl2g(x_l) * l2g_gate + out_g2g + F.conv2d(out_g2g, self.sepc_trans_alpha, stride=self.convg2g.stride)
        

        return out_xl, out_xg

class TsaTwoFFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(TsaTwoFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

        self.local_alpha = nn.Parameter(torch.ones(out_cl, in_cl, 1, 1))
        self.local_alpha.requires_grad = True
        self.global_alpha = nn.Parameter(torch.ones(out_cg, in_cg, 1, 1))
        self.global_alpha.requires_grad = True
        

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
            out_xl = out_xl + F.conv2d(x_l, self.local_alpha, stride=self.convl2l.stride)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
            out_xg = out_xg + F.conv2d(x_g, self.global_alpha, stride=self.convg2g.stride)
        

        return out_xl, out_xg

class TsaFFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(TsaFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

        self.convl2l_alpha = nn.Parameter(torch.ones(out_cl, in_cl, 1, 1))
        self.convl2l_alpha.requires_grad = True
        self.convl2g_alpha = nn.Parameter(torch.ones(out_cg, in_cl, 1, 1))
        self.convl2g_alpha.requires_grad = True
        self.convg2l_alpha = nn.Parameter(torch.ones(out_cl, in_cg, 1, 1))
        self.convg2l_alpha.requires_grad = True
        self.convg2g_alpha = nn.Parameter(torch.ones(out_cg, in_cg, 1, 1))
        self.convg2g_alpha.requires_grad = True

        

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            tsa_convl2l = self.convl2l(x_l) + F.conv2d(x_l, self.convl2l_alpha, stride=self.convl2l.stride)
            tsa_convg2l = self.convg2l(x_g) + F.conv2d(x_g, self.convg2l_alpha, stride=self.convg2l.stride)
            out_xl = tsa_convl2l + tsa_convg2l * g2l_gate
        if self.ratio_gout != 0:
            tsa_convl2g = self.convl2g(x_l) + F.conv2d(x_l, self.convl2g_alpha, stride=self.convl2g.stride)
            tsa_convg2g = self.convg2g(x_g) + F.conv2d(x_g, self.convg2g_alpha, stride=self.convg2g.stride)
            out_xg = tsa_convl2g * l2g_gate + tsa_convg2g

        return out_xl, out_xg

class TsaG2GL2LFFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(TsaG2GL2LFFC_BN_ACT, self).__init__()
        self.ffc = TsaG2GL2LFFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class TsaG2GL2GFFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(TsaG2GL2GFFC_BN_ACT, self).__init__()
        self.ffc = TsaG2GL2GFFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class TsaG2GFFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(TsaG2GFFC_BN_ACT, self).__init__()
        self.ffc = TsaG2GFFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class TsaOneFFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(TsaOneFFC_BN_ACT, self).__init__()
        self.ffc = TsaOneFFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class TsaTwoFFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(TsaTwoFFC_BN_ACT, self).__init__()
        self.ffc = TsaTwoFFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class TsaFFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(TsaFFC_BN_ACT, self).__init__()
        self.ffc = TsaFFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class TsaG2GL2LFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = TsaG2GL2LFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = TsaG2GL2LFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out

class TsaG2GL2GFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = TsaG2GL2GFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = TsaG2GL2GFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out

class TsaG2GFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = TsaG2GFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = TsaG2GFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class TsaOneFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = TsaOneFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = TsaOneFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out

class TsaTwoFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = TsaTwoFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = TsaTwoFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out

class TsaFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = TsaFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = TsaFFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out

class pa(nn.Module):
    """ 
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.pa_weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.pa_weight.requires_grad = False # Adjust during training

    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.pa_weight.to(x.device))
        return x

class TsaG2GL2LFFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TsaG2GL2LFFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [TsaG2GL2LFFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

        # self.pa = pa(feat_dim=4)

    def forward(self, input, pa=False):
        # if pa:
        #     input = self.pa(input)
        return self.model(input)

class TsaG2GL2GFFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TsaG2GL2GFFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [TsaG2GL2GFFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

        self.pa = pa(feat_dim=4)

    def forward(self, input, pa=False):
        if pa:
            input = self.pa(input)
        return self.model(input)

class TsaG2GFFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TsaG2GFFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [TsaG2GFFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

        # self.pa = pa(feat_dim=4)

    def forward(self, input, pa=False):
        # if pa:
        #     input = self.pa(input)
        return self.model(input)


class TsaOneFFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TsaOneFFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [TsaOneFFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

        self.pa = pa(feat_dim=4)

    def forward(self, input, pa=False):
        if pa:
            input = self.pa(input)
        return self.model(input)


class TsaTwoFFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TsaTwoFFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [TsaTwoFFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

        self.pa = pa(feat_dim=4)

    def forward(self, input, pa=False):
        if pa:
            input = self.pa(input)
        return self.model(input)

class TsaFourFFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = TsaFFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [TsaFFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

        self.pa = pa(feat_dim=4)

    def forward(self, input, pa=False):
        if pa:
            input = self.pa(input)
        return self.model(input)


class SingleFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
        
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight.data.fill_(0.001)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        # x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + self.fuse_weight*x_l, id_g + self.fuse_weight*x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out

class SimpleFFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(SimpleFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        # module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        # self.convl2g = module(in_cl, out_cg, kernel_size,
        #                       stride, padding, dilation, groups, bias, padding_mode=padding_type)
        # module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        # self.convg2l = module(in_cg, out_cl, kernel_size,
        #                       stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l)
        if self.ratio_gout != 0:
            out_xg = self.convg2g(x_g)

        return out_xl, out_xg


class SimpleBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = SimpleFFC(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
        
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight.data.fill_(0.001)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        # x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + self.fuse_weight*x_l, id_g + self.fuse_weight*x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class FFCResNetFixAddFFCGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

        additional_model = []
        ### additional resnet blocks
        for i in range(n_blocks):
            cur_resblock = SimpleBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            additional_model += [cur_resblock]
        self.additional_model = nn.Sequential(*additional_model)

    def forward(self, input):
        for block in self.model[:5]:
            input = block(input)
        for number, block in enumerate(self.model[5:-13]):
            input = self.additional_model[number](block(input))
        for block in self.model[-13:]:
            input = block(input)
        
        return input


def calc_mean_var(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_var = feat_var.view(N, C, 1, 1)
    # feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_var


def FSM(x, y, alpha, eps=1e-5, only_some_imgs=0):
    # x_mu, x_var = tf.nn.moments(x, axes=[1,2], keepdims=True) # Nx1x1xC
    # y_mu, y_var = tf.nn.moments(y, axes=[1,2], keepdims=True) # Nx1x1xC

    x_mu, x_var = calc_mean_var(x) 
    y_mu, y_var = calc_mean_var(y) 
    
    # normalize
    x_norm = (x - x_mu) / torch.sqrt(x_var + eps)
    
    # de-normalize
    x_fsm = x_norm * torch.sqrt(y_var + eps) + y_mu
    
    # combine
    x_mix = alpha * x + (1 - alpha) * x_fsm
    
    if only_some_imgs>0:
        x_mix = torch.cat([x[:-only_some_imgs,:,:,:],x_mix[-only_some_imgs:,:,:,:]],dim=0)

    return x_mix # NxHxWxC

class FFC_FSMR_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_FSMR_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x, use_fsmr=False, shuffle_indices=None, alpha=0, only_x_g=False, only_some_imgs=0):
        x_l, x_g = self.ffc(x)
        if use_fsmr:
            if torch.is_tensor(x_g):
                if only_x_g:
                    y_g = x_g[shuffle_indices]
                    x_g = FSM(x_g, y_g, alpha, only_some_imgs=only_some_imgs)
                else:
                    y_l, y_g =  x_l[shuffle_indices], x_g[shuffle_indices]
                    x_l, x_g = FSM(x_l, y_l, alpha, only_some_imgs=only_some_imgs), FSM(x_g, y_g, alpha, only_some_imgs=only_some_imgs)
            else:
                y_l =  x_l[shuffle_indices]
                x_l = FSM(x_l, y_l, alpha, only_some_imgs=only_some_imgs)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFC_FSMR_ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_FSMR_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_FSMR_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x, use_fsmr=False, shuffle_indices=None, alpha=0, only_x_g=False, only_some_imgs=0):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g), use_fsmr=use_fsmr, shuffle_indices=shuffle_indices, alpha=alpha, only_x_g=only_x_g, only_some_imgs=only_some_imgs)
        x_l, x_g = self.conv2((x_l, x_g), use_fsmr=use_fsmr, shuffle_indices=shuffle_indices, alpha=alpha, only_x_g=only_x_g, only_some_imgs=only_some_imgs)

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class FFCResNetFSMRGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_FSMR_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_FSMR_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFC_FSMR_ResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input, use_fsmr=False, fsmr_blocks=None, only_x_g=False, only_some_imgs=0, feats_add=False, feats_add_both=False):
        if use_fsmr:
            axis = 0
            shuffle_indices = torch.randperm(input.shape[axis])
            # shuffle_indices = torch.random.shuffle(indices)
            alpha = torch.FloatTensor(1).uniform_(0,1).cuda()
            block_number = 0
            feat_number = 0
            feats = []
            for block in self.model[:fsmr_blocks[0]]:  ## to last block
                input = block(input)
                if (block_number>0 and block_number<4) and feats_add:
                    feats.append(input[0].clone())
                    feat_number += 1
                if block_number>14 and block.__class__.__name__=='ConvTranspose2d' and feats_add:
                    input = input + feats[feat_number-1]
                    feat_number -= 1
                block_number += 1
                # if isinstance(input, tuple):
                #     if torch.is_tensor(input[0]):
                #         print(input[0].shape)
                #     if torch.is_tensor(input[1]):
                #         print(input[1].shape)
                # else:
                #     print(input.shape)


            for block in range(fsmr_blocks[0],fsmr_blocks[1]+1):  ## need fsmr block
                block_class_name = self.model[block].__class__.__name__
                if block_class_name == 'FFC_FSMR_BN_ACT' or block_class_name == 'FFC_FSMR_ResnetBlock':
                    input = self.model[block](input, use_fsmr=True, shuffle_indices=shuffle_indices, alpha=alpha, only_x_g=only_x_g, only_some_imgs=only_some_imgs)
                elif block_class_name == 'ConvTranspose2d' or block_class_name == 'Conv2d':
                    input = self.model[block](input)
                    y_input =  input[shuffle_indices]
                    input = FSM(input, y_input, alpha)
                else:
                    input = self.model[block](input)
                # if isinstance(input, tuple):
                #     if torch.is_tensor(input[0]):
                #         print(input[0].shape,end='         ')
                #     if torch.is_tensor(input[1]):
                #         print(input[1].shape)
                # else:
                #     print(input.shape)
                if (block_number>0 and block_number<4) and feats_add:
                    feats.append(input[0].clone())
                    feat_number += 1
                if block_number>14 and block.__class__.__name__=='ConvTranspose2d' and feats_add:
                    input = input + feats[feat_number-1]
                    feat_number -= 1
                block_number += 1
                
            for block in self.model[fsmr_blocks[1]+1:]:  ## to last block
                input = block(input)
                # if isinstance(input, tuple):
                #     if torch.is_tensor(input[0]):
                #         print(input[0].shape,end='         ')
                #     if torch.is_tensor(input[1]):
                #         print(input[1].shape)
                # else:
                #     print(input.shape)
                if (block_number>0 and block_number<4) and feats_add:
                    feats.append(input[0].clone())
                    feat_number += 1
                if block_number>14 and block.__class__.__name__=='ConvTranspose2d' and feats_add:
                    input = input + feats[feat_number-1]
                    feat_number -= 1
                block_number += 1
            return input
        elif feats_add_both:
            block_number = 0
            feat_number = 0
            feats = []
            for block in self.model:  
                input = block(input)
                if (block_number>0 and block_number<4):
                    feats.append(input[0].clone())
                    feat_number += 1
                if block_number>14 and block.__class__.__name__=='ConvTranspose2d':
                    input = input + feats[feat_number-1]
                    feat_number -= 1
                block_number += 1
            return input
        else:
            return self.model(input)


class FFCResNetGeneratorDropout(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={},
                 p_dropout=0):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)
        self.dropout = nn.Dropout(p = p_dropout)

    def forward(self, input):
        for block in self.model[:5]:
            input = block(input)
        input = self.model[5](input)
        input = (self.dropout(input[0]), self.dropout(input[1]))
        # for block in self.model[5:-13]:
        #     input = block(input)
        #     input = (self.dropout(input[0]), self.dropout(input[1]))

        for block in self.model[6:]:
            input = block(input)
        return input

class FFCResNetGeneratorSmall(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={},
                 n_blocks_small = 3):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

        small_model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            small_model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks_small):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            small_model += [cur_resblock]

        small_model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            small_model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            small_model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        small_model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            small_model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.small_model = nn.Sequential(*small_model)


    def forward(self, input, training):
        if training:
            input_small = input

            for block in self.model[:-13]:
                input = block(input)
            acti_model = self.model[-13](input)

            for block in self.small_model[:-13]:
                input_small = block(input_small)
            input_small = self.small_model[-13](input_small)
            acti_small_model = input_small
            for block in self.small_model[-12:]:
                input_small = block(input_small)

            return acti_model, acti_small_model, input_small
            
        else:
            return self.small_model(input)

class FFCResNetGeneratorMaxChangeAdd(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        max_change_blocks = []
        ###  resnet blocks
        for i in range(3):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            max_change_blocks += [cur_resblock]
        self.max_change_blocks = nn.Sequential(*max_change_blocks)

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        
        for block in self.model[:5]:
            input = block(input)
        
        input = self.model[5](self.max_change_blocks[0](input))

        for block in self.model[6:20]:
            input = block(input)
        
        input = self.model[20](self.max_change_blocks[1](input))
        input = self.model[21](input)
        input = self.model[22](self.max_change_blocks[2](input))

        for block in self.model[23:]:
            input = block(input)

        return input


class FFCResNetGeneratorSimpleAdd(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        pretrained_blocks = []
        ###  resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            pretrained_blocks += [cur_resblock]
        self.pretrained_blocks = nn.Sequential(*pretrained_blocks)

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        
        for block in self.model[:5]:
            input = block(input)
        
        for block_finetune, blcok_pretrain in zip(self.model[5:-13], self.pretrained_blocks):
            input = block_finetune(blcok_pretrain(input))
            
        for block in self.model[-13:]:
            input = block(input)
        return input


class FFCResNetGeneratorSpottune(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        pretrained_blocks = []
        ###  resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            pretrained_blocks += [cur_resblock]
        self.pretrained_blocks = nn.Sequential(*pretrained_blocks)

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input, policy=None):
        if policy is not None:
            t = 0
            for block in self.model[:5]:
                input = block(input)
            
            for block_finetune, blcok_pretrain in zip(self.model[5:-13], self.pretrained_blocks):
                action = policy[:,t].contiguous()
                action_mask = action.float().view(-1,1,1,1)

                output = block_finetune(input)
                output_pretrain = blcok_pretrain(input)

                input = (output[0]*(1-action_mask) + output_pretrain[0]*action_mask, 
                    output[1]*(1-action_mask) + output_pretrain[1]*action_mask)
                t += 1

            for block in self.model[-13:]:
                input = block(input)
            return input
        else:
            return self.model(input)


class FFCResNetGeneratorSecondStage(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, alpha=0, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        self.alpha = alpha # for fade in

        # --------------------------  first stage model ------------------------------
        # --------------------------  f     s     model ------------------------------
        fs_model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            fs_model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            fs_model += [cur_resblock]

        fs_model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            fs_model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            fs_model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        fs_model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            fs_model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.fs_model = nn.Sequential(*fs_model)

        # --------------------------  second stage model ------------------------------

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input, input_with_no_mask):
        if self.alpha == 0:
            return self.fs_model(input_with_no_mask)
        elif self.alpha>0 and self.alpha<1:
            fade_number = [14]
            for number, (fs_block, model_block) in enumerate(zip(self.fs_model, self.model)):
                # print(number, (fs_block, model_block))
                input_with_no_mask = fs_block(input_with_no_mask)
                input = model_block(input)
                if number in fade_number:
                    input = self.alpha * input + (1 - self.alpha) * input_with_no_mask
            return input
        elif self.alpha==1:
            return self.model(input)
        else:
            raise ValueError('alpha must >0 or <1')

        # return self.model(input)


class FFCNLayerDiscriminator(BaseDiscriminator):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, max_features=512,
                 init_conv_kwargs={}, conv_kwargs={}):
        super().__init__()
        self.n_layers = n_layers

        def _act_ctor(inplace=True):
            return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)

        kw = 3
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[FFC_BN_ACT(input_nc, ndf, kernel_size=kw, padding=padw, norm_layer=norm_layer,
                                activation_layer=_act_ctor, **init_conv_kwargs)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, max_features)

            cur_model = [
                FFC_BN_ACT(nf_prev, nf,
                           kernel_size=kw, stride=2, padding=padw,
                           norm_layer=norm_layer,
                           activation_layer=_act_ctor,
                           **conv_kwargs)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = [
            FFC_BN_ACT(nf_prev, nf,
                       kernel_size=kw, stride=1, padding=padw,
                       norm_layer=norm_layer,
                       activation_layer=lambda *args, **kwargs: nn.LeakyReLU(*args, negative_slope=0.2, **kwargs),
                       **conv_kwargs),
            ConcatTupleLayer()
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        feats = []
        for out in act[:-1]:
            if isinstance(out, tuple):
                if torch.is_tensor(out[1]):
                    out = torch.cat(out, dim=1)
                else:
                    out = out[0]
            feats.append(out)
        return act[-1], feats

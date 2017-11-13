import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from util.util import *
from math import floor
import pdb
###############################################################################
# Functions
###############################################################################



# TODO, look into init functions
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_zeros(m):
    classname = m.__class__.__name__
    print(classname)
    # pdb.set_trace()
    if classname.find('Conv') != -1 and classname != 'ConvLstmCell':
        # init.xavier_normal(m.weight.data, gain=1)
        init.uniform(m.weight.data, 0.0, 0.02)
        init.constant(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()
        # pdb.set_trace()
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
        init.constant(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()


def weights_init_mcnet(m):
    classname = m.__class__.__name__
    print(classname)
    # pdb.set_trace()
    if classname.find('Conv') != -1 and classname != 'ConvLstmCell':
        init.xavier_normal(m.weight.data, gain=1)
        # init.constant(m.weight.data, 0.0)
        init.constant(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()
        # pdb.set_trace()
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
        init.constant(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()



def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'mcnet':
        net.apply(weights_init_mcnet)
    elif init_type == 'zeros':
        net.apply(weights_init_zeros)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def define_motion_enc(gf_dim, init_type="mcnet", gpu_ids=[]):
    # motion_enc = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    motion_enc = MotionEnc(gf_dim, gpu_ids)

    if len(gpu_ids) > 0:
        motion_enc.cuda(device_id=gpu_ids[0])

    # pdb.set_trace()

    init_weights(motion_enc, init_type=init_type)
    return motion_enc


def define_content_enc(c_dim, gf_dim, init_type="mcnet", gpu_ids=[]):
    # motion_enc = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    content_enc = ContentEnc(c_dim, gf_dim, gpu_ids)

    if len(gpu_ids) > 0:
        content_enc.cuda(device_id=gpu_ids[0])

    init_weights(content_enc, init_type=init_type)
    return content_enc


def define_comb_layers(gf_dim, init_type="mcnet", gpu_ids=[]):
    # motion_enc = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    comb_layers = CombLayers(gf_dim, gpu_ids)

    if len(gpu_ids) > 0:
        comb_layers.cuda(device_id=gpu_ids[0])

    init_weights(comb_layers, init_type=init_type)
    return comb_layers


def define_residual(out_dim, init_type="mcnet", gpu_ids=[]):
    # motion_enc = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    residual = Residual(out_dim, gpu_ids)

    if len(gpu_ids) > 0:
        residual.cuda(device_id=gpu_ids[0])

    init_weights(residual, init_type=init_type)
    return residual


def define_dec_cnn(c_dim, gf_dim, init_type="mcnet", gpu_ids=[]):
    # motion_enc = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    dec_cnn = DecCnn(c_dim, gf_dim, gpu_ids)

    if len(gpu_ids) > 0:
        dec_cnn.cuda(device_id=gpu_ids[0])

    init_weights(dec_cnn, init_type=init_type)
    return dec_cnn


def fixed_unpooling(x, gpu_ids):
    x = x.permute(0, 2, 3, 1)
    if len(gpu_ids)>0:
        out = torch.cat((x, Variable(torch.zeros(x.size())).cuda()), dim=3)
        out = torch.cat((out, Variable(torch.zeros(out.size())).cuda()), dim=2)
    else:
        out = torch.cat((x, Variable(torch.zeros(x.size()))), dim=3)
        out = torch.cat((out, Variable(torch.zeros(out.size()))), dim=2)

    sh = x.size()
    s0, s1, s2, s3 = int(sh[0]), int(sh[1]), int(sh[2]), int(sh[3])
    s1 *= 2
    s2 *= 2
    return out.view(s0, s1, s2, s3).permute(0, 3, 1, 2)


def define_discriminator(img_size, c_dim, in_num, out_num, df_dim, init_type="mcnet", gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    discriminator = Discriminator(img_size, c_dim, in_num, out_num, df_dim, gpu_ids)

    if len(gpu_ids) > 0:
        discriminator.cuda(device_id=gpu_ids[0])

    init_weights(discriminator, init_type='zeros')
    return discriminator


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_gdl(c_dim, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    gdl = GDL(c_dim, gpu_ids)

    if len(gpu_ids) > 0:
        gdl.cuda(device_id=gpu_ids[0])

    return gdl


def define_convLstm_cell(feature_size, num_features, forget_bias=1, activation=F.tanh, gpu_ids=[], init_type="mcnet", bias=True):
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    convLstm_cell = ConvLstmCell(feature_size, num_features, gpu_ids, forget_bias=forget_bias, activation=activation, bias=bias)

    if len(gpu_ids) > 0:
        convLstm_cell.cuda(device_id=gpu_ids[0])

    # pdb.set_trace()

    init_weights(convLstm_cell, init_type=init_type)
    return convLstm_cell


def define_generator(gf_dim, c_dim, feature_size, forget_bias=1, activation=F.tanh, gpu_ids=[], init_type="mcnet", bias=True):
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    generator = Generator(gf_dim, c_dim, feature_size, gpu_ids, forget_bias=forget_bias,activation=activation, bias=bias)

    if len(gpu_ids) > 0:
        generator.cuda(device_id=gpu_ids[0])

    init_weights(generator, init_type=init_type)
    return generator

##############################################################################
# Classes
##############################################################################


class MotionEnc(nn.Module):
    def __init__(self, gf_dim, gpu_ids):
        super(MotionEnc, self).__init__()
        self.gpu_ids = gpu_ids

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        conv1 = nn.Conv2d(1, gf_dim, 5, padding=2)
        relu1 = nn.ReLU()
        dyn_conv1 = [conv1, relu1]
        self.dyn_conv1 = nn.Sequential(*dyn_conv1)

        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        pool1 = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(gf_dim, gf_dim*2, 5, padding=2)
        relu2 = nn.ReLU()
        dyn_conv2 = [pool1, conv2, relu2]
        self.dyn_conv2 = nn.Sequential(*dyn_conv2)

        pool2 = nn.MaxPool2d(2)
        conv3 = nn.Conv2d(gf_dim * 2, gf_dim * 4, 7, padding=3)
        relu3 = nn.ReLU()
        dyn_conv3 = [pool2, conv3, relu3]
        self.dyn_conv3 = nn.Sequential(*dyn_conv3)

        self.pool3 = nn.MaxPool2d(2)

    def forward(self, input_diff):
        """
        input_diff: [batch_size, 1, h, w]
        res_in: a list of 3 tensors, [batch_size, gf_dim, h, w],
                   [batch_size, gf_dim*2, h/2, w/2],
                   [batch_size, gf_dim*4, h/4, w/4]
        output: [batch_size, gf_dim*4, h/8, w/8]
        """
        res_in = []

        if len(self.gpu_ids) > 0 and isinstance(input_diff.data, torch.cuda.FloatTensor):
            res_in.append(nn.parallel.data_parallel(self.dyn_conv1, input_diff, self.gpu_ids))
            res_in.append(nn.parallel.data_parallel(self.dyn_conv2, res_in[-1], self.gpu_ids))
            res_in.append(nn.parallel.data_parallel(self.dyn_conv3, res_in[-1], self.gpu_ids))
            output = nn.parallel.data_parallel(self.pool3, res_in[-1], self.gpu_ids)
            return output, res_in
        else:
            res_in.append(self.dyn_conv1(input_diff))
            res_in.append(self.dyn_conv2(res_in[-1]))
            res_in.append(self.dyn_conv3(res_in[-1]))
            output = self.pool3(res_in[-1])
            return output, res_in


class ContentEnc(nn.Module):
    def __init__(self, c_dim, gf_dim, gpu_ids):
        super(ContentEnc, self).__init__()
        self.gpu_ids = gpu_ids

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        conv1_1 = nn.Conv2d(c_dim, gf_dim, 3, padding=1)
        relu1_1 = nn.ReLU()
        conv1_2 = nn.Conv2d(gf_dim, gf_dim, 3, padding=1)
        relu1_2 = nn.ReLU()
        cont_conv1 = [conv1_1, relu1_1, conv1_2, relu1_2]
        self.cont_conv1 = nn.Sequential(*cont_conv1)

        pool1 = nn.MaxPool2d(2)
        conv2_1 = nn.Conv2d(gf_dim, gf_dim*2, 3, padding=1)
        relu2_1 = nn.ReLU()
        conv2_2 = nn.Conv2d(gf_dim*2, gf_dim*2, 3, padding=1)
        relu2_2 = nn.ReLU()
        cont_conv2 = [pool1, conv2_1, relu2_1, conv2_2, relu2_2]
        self.cont_conv2 = nn.Sequential(*cont_conv2)

        pool2 = nn.MaxPool2d(2)
        conv3_1 = nn.Conv2d(gf_dim * 2, gf_dim * 4, 3, padding=1)
        relu3_1 = nn.ReLU()
        conv3_2 = nn.Conv2d(gf_dim * 4, gf_dim * 4, 3, padding=1)
        relu3_2 = nn.ReLU()
        conv3_3 = nn.Conv2d(gf_dim * 4, gf_dim * 4, 3, padding=1)
        relu3_3 = nn.ReLU()
        cont_conv3 = [pool2, conv3_1, relu3_1, conv3_2, relu3_2, conv3_3, relu3_3]

        self.cont_conv3 = nn.Sequential(*cont_conv3)

        self.pool3 = nn.MaxPool2d(2)

    def forward(self, raw):
        """
        input_diff: [batch_size, c_dim, h, w]
        res_in: a list of 3 tensors, [batch_size, gf_dim, h, w],
                   [batch_size, gf_dim*2, h/2, w/2],
                   [batch_size, gf_dim*4, h/4, w/4]
        output: [batch_size, gf_dim*4, h/8, w/8]
        """
        res_in = []
        if len(self.gpu_ids) > 0 and isinstance(raw.data, torch.cuda.FloatTensor):
            res_in.append(nn.parallel.data_parallel(self.cont_conv1, raw, self.gpu_ids))
            res_in.append(nn.parallel.data_parallel(self.cont_conv2, res_in[-1], self.gpu_ids))
            res_in.append(nn.parallel.data_parallel(self.cont_conv3, res_in[-1], self.gpu_ids))
            output = nn.parallel.data_parallel(self.pool3, res_in[-1], self.gpu_ids)
            return output, res_in
        else:
            res_in.append(self.cont_conv1(raw))
            res_in.append(self.cont_conv2(res_in[-1]))
            res_in.append(self.cont_conv3(res_in[-1]))
            output = self.pool3(res_in[-1])
            return output, res_in


class CombLayers(nn.Module):
    def __init__(self, gf_dim, gpu_ids):
        super(CombLayers, self).__init__()
        self.gpu_ids = gpu_ids

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        conv1 = nn.Conv2d(gf_dim * 8, gf_dim * 4, 3, padding=1)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, padding=1)
        relu2 = nn.ReLU()
        conv3 = nn.Conv2d(gf_dim * 2, gf_dim * 4, 3, padding=1)
        relu3 = nn.ReLU()
        h_comb = [conv1, relu1, conv2, relu2, conv3, relu3]
        self.h_comb = nn.Sequential(*h_comb)

    def forward(self, h_dyn, h_cont):
        input = torch.cat((h_dyn, h_cont), dim=1)
        if len(self.gpu_ids) > 0 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.h_comb, input, self.gpu_ids)
        else:
            return self.h_comb(input)


class Residual(nn.Module):
    def __init__(self, out_dim, gpu_ids):
        super(Residual, self).__init__()
        self.gpu_ids = gpu_ids

        in_dim = 2 * out_dim
        conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        res = [conv1, relu1, conv2]
        self.res = nn.Sequential(*res)

    def forward(self, input_dyn, input_cont):
        input = torch.cat((input_dyn, input_cont), dim=1)
        if len(self.gpu_ids)>0 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.res, input, self.gpu_ids)
        else:
            return self.res(input)


class DecCnn(nn.Module):
    def __init__(self, c_dim, gf_dim, gpu_ids):
        super(DecCnn, self).__init__()
        self.gpu_ids = gpu_ids

        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        deconv3_3 = nn.ConvTranspose2d(gf_dim * 4, gf_dim * 4, 3, padding=1)
        relu3_3 = nn.ReLU()
        deconv3_2 = nn.ConvTranspose2d(gf_dim * 4, gf_dim * 4, 3, padding=1)
        relu3_2 = nn.ReLU()
        deconv3_1 = nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, 3, padding=1)
        relu3_1 = nn.ReLU()
        dec3 = [deconv3_3, relu3_3, deconv3_2, relu3_2, deconv3_1, relu3_1]
        self.dec3 = nn.Sequential(*dec3)

        deconv2_2 = nn.ConvTranspose2d(gf_dim * 2, gf_dim * 2, 3, padding=1)
        relu2_2 = nn.ReLU()
        deconv2_1 = nn.ConvTranspose2d(gf_dim * 2, gf_dim, 3, padding=1)
        relu2_1 = nn.ReLU()
        dec2 = [deconv2_2, relu2_2, deconv2_1, relu2_1]
        self.dec2 = nn.Sequential(*dec2)

        deconv1_2 = nn.ConvTranspose2d(gf_dim, gf_dim, 3, padding=1)
        relu1_2 = nn.ReLU()
        deconv1_1 = nn.ConvTranspose2d(gf_dim, c_dim, 3, padding=1)
        tanh1_1 = nn.Tanh()
        dec1 = [deconv1_2, relu1_2, deconv1_1, tanh1_1]
        self.dec1 = nn.Sequential(*dec1)

    def forward(self, res1, res2, res3, comb):
        if len(self.gpu_ids)>0 and isinstance(res1.data, torch.cuda.FloatTensor):
            input3 = fixed_unpooling(comb, self.gpu_ids) + res3
            dec3_out = nn.parallel.data_parallel(self.dec3, input3, self.gpu_ids)
            input2 = fixed_unpooling(dec3_out, self.gpu_ids) + res2
            dec2_out = nn.parallel.data_parallel(self.dec2, input2, self.gpu_ids)
            input1 = fixed_unpooling(dec2_out, self.gpu_ids) + res1
            dec1_out = nn.parallel.data_parallel(self.dec1, input1, self.gpu_ids)
            return dec1_out
        else:
            input3 = fixed_unpooling(comb, self.gpu_ids) + res3
            dec3_out = self.dec3(input3)
            input2 = fixed_unpooling(dec3_out, self.gpu_ids) + res2
            dec2_out = self.dec2(input2)
            input1 = fixed_unpooling(dec2_out, self.gpu_ids) + res1
            dec1_out = self.dec1(input1)
            return dec1_out


class Discriminator(nn.Module):
    def __init__(self, img_size, c_dim, in_num, out_num, df_dim, gpu_ids):
        super(Discriminator, self).__init__()
        self.gpu_ids = gpu_ids
        h, w = img_size[0], img_size[1]

        conv0 = nn.Conv2d(c_dim * (in_num + out_num), df_dim, 4, stride=2, padding=1)
        h = floor((h + 2*1 - 4)/2 + 1)
        w = floor((w + 2*1 - 4)/2 + 1)
        # torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        lrelu0 = nn.LeakyReLU(0.2)

        conv1 = nn.Conv2d(df_dim, df_dim * 2, 4, stride=2, padding=1)
        h = floor((h + 2 * 1 - 4) / 2 + 1)
        w = floor((w + 2 * 1 - 4) / 2 + 1)
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
        # bn1 = nn.BatchNorm2d(df_dim * 2, eps=0.001, momentum=0.1)
        bn1 = nn.BatchNorm2d(df_dim * 2)
        lrelu1 = nn.LeakyReLU(0.2)

        conv2 = nn.Conv2d(df_dim * 2, df_dim * 4, 4, stride=2, padding=1)
        h = floor((h + 2 * 1 - 4) / 2 + 1)
        w = floor((w + 2 * 1 - 4) / 2 + 1)
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
        # bn2 = nn.BatchNorm2d(df_dim * 4, eps=0.001, momentum=0.1)
        bn2 = nn.BatchNorm2d(df_dim * 4)
        lrelu2 = nn.LeakyReLU(0.2)

        conv3 = nn.Conv2d(df_dim * 4, df_dim * 8, 4, stride=2, padding=1)
        h = floor((h + 2 * 1 - 4) / 2 + 1)
        w = floor((w + 2 * 1 - 4) / 2 + 1)
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
        # bn3 = nn.BatchNorm2d(df_dim * 8, eps=0.001, momentum=0.1)
        bn3 = nn.BatchNorm2d(df_dim * 8)
        lrelu3 = nn.LeakyReLU(0.2)

        D = [conv0, lrelu0, conv1, bn1, lrelu1, conv2, bn2, lrelu2, conv3, bn3, lrelu3]
        self.D = nn.Sequential(*D)

        in_features = int(h * w * df_dim * 8)

        # torch.nn.Linear(in_features, out_features, bias=True)
        self.linear = nn.Linear(in_features, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, batch_size):
        if len(self.gpu_ids)>0 and isinstance(input.data, torch.cuda.FloatTensor):
            D_output = nn.parallel.data_parallel(self.D, input, self.gpu_ids)
            D_output = D_output.view(batch_size, -1)
            h = nn.parallel.data_parallel(self.linear, D_output, self.gpu_ids)
            h_sigmoid = nn.parallel.data_parallel(self.sigmoid, h, self.gpu_ids)
            return h_sigmoid, h
        else:
            D_output = self.D(input)
            D_output = D_output.view(batch_size, -1)
            h = self.linear(D_output)
            h_sigmoid = self.sigmoid(h)
            return h_sigmoid, h


class GDL(nn.Module):
    def __init__(self, c_dim, gpu_ids):
        super(GDL, self).__init__()
        self.gpu_ids = gpu_ids
        self.loss = nn.L1Loss()
        a = np.array([[-1, 1]])
        b = np.array([[1], [-1]])
        self.filter_w = np.zeros([c_dim, c_dim, 1, 2])
        self.filter_h = np.zeros([c_dim, c_dim, 2, 1])
        for i in range(c_dim):
            self.filter_w[i, i, :, :] = a
            self.filter_h[i, i, :, :] = b
        

    def __call__(self, output, target):
        # pdb.set_trace()
        if len(self.gpu_ids) > 0:
            filter_w = Variable(torch.from_numpy(self.filter_w).float().cuda())
            filter_h = Variable(torch.from_numpy(self.filter_h).float().cuda())
        else:
            filter_w = Variable(torch.from_numpy(self.filter_w).float())
            filter_h = Variable(torch.from_numpy(self.filter_h).float())
        output_w = F.conv2d(output, filter_w, padding=(0, 1))
        output_h = F.conv2d(output, filter_h, padding=(1, 0))
        target_w = F.conv2d(target, filter_w, padding=(0, 1))
        target_h = F.conv2d(target, filter_h, padding=(1, 0))
        return self.loss(output_w, target_w) + self.loss(output_h, target_h)


class ConvLstmCell(nn.Module):
    def __init__(self, feature_size, num_features, gpu_ids, forget_bias=1, activation=F.tanh, bias=True):
        super(ConvLstmCell, self).__init__()
        self.gpu_ids = gpu_ids

        self.feature_size = feature_size
        self.num_features = num_features
        self.forget_bias = forget_bias
        self.activation = activation

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(num_features * 2, num_features*4, feature_size, padding=(feature_size-1)/2, bias=bias)

    def forward(self, input, state):
        c, h = torch.chunk(state, 2, dim=1)
        conv_input = torch.cat((input, h) , dim=1)
        if len(self.gpu_ids)>0 and isinstance(input.data, torch.cuda.FloatTensor):
            conv_output = nn.parallel.data_parallel(self.conv, conv_input)
        else:
            conv_output = self.conv(conv_input)
        (i, j, f, o) = torch.chunk(conv_output, 4, dim=1)
        new_c = c * F.sigmoid(f+self.forget_bias) + F.sigmoid(i) * self.activation(j)
        new_h = self.activation(new_c) * F.sigmoid(o)
        new_state = torch.cat((new_c, new_h), dim=1)
        return new_h, new_state

class Generator(nn.Module):
    def __init__(self, gf_dim, c_dim, feature_size, gpu_ids, forget_bias=1, activation=F.tanh, bias=True):
        super(Generator, self).__init__()
        self.gpu_ids = gpu_ids
        self.c_dim = c_dim

        self.motion_enc = define_motion_enc(gf_dim, gpu_ids=self.gpu_ids)  # an object of class MotionEnc
        #  define_convLstm_cell(feature_size, num_features, forget_bias=1, activation=F.tanh, gpu_ids=[], init_type="mcnet", bias=True)
        self.convLstm_cell = define_convLstm_cell(feature_size, 4 * gf_dim, gpu_ids=self.gpu_ids, forget_bias=forget_bias, activation=activation, bias=bias)
        self.content_enc = define_content_enc(c_dim, gf_dim, gpu_ids=self.gpu_ids)
        self.comb_layers = define_comb_layers(gf_dim, gpu_ids=self.gpu_ids)
        self.residual3 = define_residual(gf_dim * 4, gpu_ids=self.gpu_ids)
        self.residual2 = define_residual(gf_dim * 2, gpu_ids=self.gpu_ids)
        self.residual1 = define_residual(gf_dim * 1, gpu_ids=self.gpu_ids)
        self.dec_cnn = define_dec_cnn(c_dim, gf_dim, gpu_ids=self.gpu_ids)

    def forward(self, K, T, state, batch_size, image_size, diff_in, targets):
        for t in range(K-1):
            enc_h, res_m = self.motion_enc.forward(diff_in[t])
            h_dyn, state = self.convLstm_cell.forward(enc_h, state)

        xt = targets[K - 1]

        pred = []
        for t in range(T):
            if t > 0:
                enc_h, res_m = self.motion_enc.forward(diff_in[-1])
                h_dyn, state = self.convLstm_cell.forward(enc_h, state)
            h_cont, res_c = self.content_enc.forward(xt)
            h_tpl = self.comb_layers.forward(h_dyn, h_cont)
            res_1 = self.residual1.forward(res_m[0], res_c[0])
            res_2 = self.residual2.forward(res_m[1], res_c[1])
            res_3 = self.residual3.forward(res_m[2], res_c[2])
            x_hat = self.dec_cnn.forward(res_1, res_2, res_3, h_tpl)

            if self.c_dim == 3:
                x_hat_gray = bgr2gray(inverse_transform(x_hat))
                xt_gray = bgr2gray(inverse_transform(xt))
            else:
                x_hat_gray = inverse_transform(x_hat)
                xt_gray = inverse_transform(xt)

            diff_in.append(x_hat_gray - xt_gray)
            xt = x_hat
            pred.append(x_hat.view(batch_size, self.c_dim, image_size[0], image_size[1]))

        return pred

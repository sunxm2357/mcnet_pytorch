import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class McnetModel(BaseModel):
    def name(self):
        return "McnetModel"

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.is_train = opt.is_train

        # define tensor
        self.K = opt.K
        self.T = opt.T
        self.opt = opt
        self.targets = [] # first K-1 are diff, the last one is raw
        for i in range(self.K+self.T):
            self.inputs.append(self.Tensor(opt.batch_size, opt.c_dim, opt.image_size[0], opt.image_size[1]))

        self.diff_in = []
        for i in range(self.K-1):
            self.targets.append(self.Tensor(opt.batch_size, 1, opt.image_size[0], opt.image_size[1]))

        # define submodules in G
        self.generator = networks.define_generator(opt.gf_dim, opt.c_dim, 3, self.gpu_ids)
        # self.motion_enc = networks.define_motion_enc(opt.gf_dim, self.gpu_ids) # an object of class MotionEnc
        # self.convLstm_cell = networks.define_convLstm_cell(3, 4 * opt.gf_dim, self.gpu_ids)
        # self.content_enc=networks.define_content_enc(opt.c_dim, opt.gf_dim, self.gpu_ids)
        # self.comb_layers = networks.define_comb_layers(opt.gf_dim, self.gpu_ids)
        # self.residual3 = networks.define_residual(opt.gf_dim*4, self.gpu_ids)
        # self.residual2 = networks.define_residual(opt.gf_dim*2, self.gpu_ids)
        # self.residual1 = networks.define_residual(opt.gf_dim*1, self.gpu_ids)
        # self.dec_cnn = networks.define_dec_cnn(opt.gf_dim, self.gpu_ids)


        if self.is_train():
            self.discriminator = networks.define_discriminator(opt.image_size, opt.c_dim, in_num, out_num, opt.df_dim, self.gpu_ids)

        # load pretrained model
        if not self.isTrain or opt.continue_train:
            self.load_network(self.generator, 'generator', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.discriminator, 'discriminator', opt.which_epoch)

        if self.is_train:
            # define loss
            self.loss_d = torch.nn.CrossEntropyLoss()
            self.loss_Lp = torch.nn.MSELoss()
            self.loss_gdl = networks.define_gdl(self.gpu_ids)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            # netG_parameters = list(self.motion_enc.parameters()) + list(self.convLstm.parameters()) + list(self.content_enc)\
            #             + list(self.comb_layers.parameters()) + list(self.comb_layers.parameters()) + list(self.residual13.parameters())\
            #             + list(self.residual2.parameters()) + list(self.residual1.parameters()) + list(self.dec_cnn.parameters())
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999)) # TODO, check the lr and beta of Mcnet train, beta1 =0.5
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # schedular is used for lr decay
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            self.updateD = True
            self.updateG = True

    def set_inputs(self, input): # TODO: changing after finishing dataloader
        targets = torch.from_numpy(input["targets"]) # shape[-1] = K+T
        diff_in = torch.from_numpy(input["diff_in"]) # shape[-1] = K-1
        #  if opt.task == "infill":
        #     future = torch.from_numpy(input["future"])
        #     diff_future = torch.from_numpy(input["diff_future"])
        #     self.input_G_future = []
        self.diff_in = []
        self.targets = []
        for i in range(self.K - 1):
            self.diff_in.append(Variable(diff_in[:, :, :, :, i]))

        for i in range(self.K + self.T):
            self.targets.append(Variable(Variable(targets[:, :, :, :, i])))

    def forward(self):
        state = Variable(torch.zeros(self.opt.batch_size, 512, self.opt.image_size[0]/8, self.opt.image_size[1]/8))
        self.pred = self.generator.forward(self.K, self.T, state, self.opt.batch_size, self.opt.image_size, self.diff_in, self.targets)
        # # Encoder
        # for t in range(self.K-1):
        #     enc_h, res_m = self.motion_enc.foward(self.diff_in[t])
        #     h_dyn, state = self.convLstm_cell.forward(enc_h, state)
        #
        # xt = self.targets[self.K - 1]
        # self.pred = []
        # for t in range(self.T):
        #     if t > 0:
        #         enc_h, res_m = self.motion_enc.foward(self.diff_in[-1])
        #         h_dyn, state = self.convLstm_cell.forward(enc_h, state)
        #     h_cont, res_c = self.content_enc.forward(xt)
        #     h_tpl = self.comb_layers.forward(h_dyn, h_cont)
        #     res_1 = self.residual1.forward(res_m[0], res_c[0])
        #     res_2 = self.residual2.forward(res_m[1], res_c[1])
        #     res_3 = self.residual3.forward(res_m[2], res_c[2])
        #     x_hat = self.dec_cnn.forward(res_1, res_2, res_3, h_tpl)
        #
        #     if self.opt.c_dim == 3:
        #         x_hat_gray = bgr2gray(inverse_transform(x_hat))
        #         xt_gray = bgr2gray(inverse_transform(xt))
        #     else:
        #         x_hat_gray = inverse_transform(x_hat)
        #         xt_gray = inverse_transform(self.targets[self.K - 1] + t)
        #
        #     self.diff_in.append(x_hat_gray - xt_gray)
        #     xt = x_hat
        #     self.pred.append(x_hat.view(self.opt.batch_size, self.opt.c_dim, self.opt.image_size[0], self.opt.image_size[1]))

    def backward_D(self):
        # fake
        input_fake = torch.cat(self.targets[:self.K] + self.pred, dim=1)
        h_sigmoid, h = self.discriminator.forward(input_fake, self.opt.batch_size)
        labels = Variable(torch.zeros_like(h))
        self.loss_d_fake = self.loss_d(h_sigmoid, labels)

        # real
        input_real = self.targets
        h_sigmoid_, h_ = self.discriminator.forward(input_real, self.opt.batch_size)
        labels = Variable(torch.ones_like(h_))
        self.loss_d_real = self.loss_d(h_sigmoid_, labels)


        self.loss_D = self.loss_d_fake + self.loss_d_real

        self.loss_D.backward()

    def backward_G(self):
        input_fake = torch.cat(self.targets[:self.K] + self.pred, dim=1)
        h_sigmoid, h = self.discriminator.forward(input_fake, self.opt.batch_size)
        labels = Variable(torch.ones_like(h))
        self.L_GAN = self.loss_d(h_sigmoid, labels)

        outputs = networks.inverse_transform(torch.cat(self.pred, dim=0))
        targets = networks.inverse_transform(torch.cat(self.targets[self.K:], dim=0))
        self.Lp = self.loss_Lp(outputs, targets)
        self.gdl = self.loss_gdl(outputs, targets)

        self.loss_G = self.opt.alpha * (self.Lp + self.gdl) + self.opt.beta * self.L_GAN

        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()

        if self.updateD:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        if self.updateG:
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

        if self.loss_d_fake < self.opt.margin or self.loss_d_real < self.opt.margin:
            self.updateD = False

        if self.loss_d_fake > (1.- self.opt.margin) or self.loss_d_real > (1.- self.opt.margin):
            self.updateG = False

        if not self.updateD and not self.updateG:
            self.updateD = True
            self.updateG = True

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.L_GAN.data[0]),
                            ('G_Lp', self.Lp.data[0]),
                            ('G_gdl', self.gdl.data[0]),
                            ('D_real', self.loss_d_real.data[0]),
                            ('D_fake', self.loss_d_fake.data[0])
                            ])


    def get_current_visuals(self):
        diff_in = util.tensorlist2imlist(self.diff_in)
        raw = util.tensor2im(self.targets[self.K-1])
        targets = util.tensorlist2imlist(self.targets)
        pred = util.tensorlist2imlist(self.pred)
        return OrderedDict([('diff_in', diff_in), ('raw', raw), ('targets', targets), ('pred', pred)])


    def save(self, label):
        # self.save_network(self.motion_enc, 'motion_enc', label, self.gpu_ids)
        # self.save_network(self.convLstm_cell, 'convLstm_cell', label, self.gpu_ids)
        # self.save_network(self.content_enc, 'content_enc', label, self.gpu_ids)
        # self.save_network(self.comb_layers, 'comb_layers', label, self.gpu_ids)
        # self.save_network(self.residual1, 'residual1', label, self.gpu_ids)
        # self.save_network(self.residual2, 'residual2', label, self.gpu_ids)
        # self.save_network(self.residual3, 'residual3', label, self.gpu_ids)
        # self.save_network(self.dec_cnn, 'dec_cnn', label, self.gpu_ids)
        self.save_network(self.generator, "generator", label, self.gpu_ids)
        self.save_network(self.discriminator, 'discriminator', label, self.gpu_ids)

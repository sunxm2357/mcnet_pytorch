import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
import pdb


class McnetModel(BaseModel):
    def name(self):
        return "McnetModel"

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.is_train = opt.is_train

        # define tensor
        self.K = opt.K
        self.T = opt.T
        self.start_epoch = opt.epoch_count

        if len(opt.gpu_ids) > 0:
            self.state = Variable(torch.zeros(self.opt.batch_size, 512, self.opt.image_size[0]/8, self.opt.image_size[1]/8).cuda(), requires_grad=False)
        else:
            self.state = Variable(torch.zeros(self.opt.batch_size, 512, self.opt.image_size[0]/8, self.opt.image_size[1]/8), requires_grad=False)


        self.targets = [] # first K-1 are diff, the last one is raw
        for i in range(self.K+self.T):
            self.targets.append(self.Tensor(opt.batch_size, opt.c_dim, opt.image_size[0], opt.image_size[1]))

        self.diff_in = []
        for i in range(self.K-1):
            self.diff_in.append(self.Tensor(opt.batch_size, 1, opt.image_size[0], opt.image_size[0]))

        # define submodules in G
        self.generator = networks.define_generator(opt.gf_dim, opt.c_dim, 3, gpu_ids=self.gpu_ids)

        self.updateD = True
        self.updateG = True

        if self.is_train:
            # define loss
            self.loss_Lp = torch.nn.MSELoss()
            self.loss_gdl = networks.define_gdl(opt.c_dim, self.gpu_ids)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            if not opt.no_adversarial:
                self.discriminator = networks.define_discriminator(opt.image_size, opt.c_dim, self.K, self.T,
                                                                   opt.df_dim, gpu_ids=self.gpu_ids)
                self.loss_d = torch.nn.BCELoss()
                self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        # load pretrained model
        if not self.is_train or opt.continue_train:
            self.load(opt.which_epoch)

        if self.is_train:
            self.optimizers.append(self.optimizer_G)
            if not opt.no_adversarial:
                self.optimizers.append(self.optimizer_D)
            # schedular is used for lr decay
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def set_inputs(self, input):
        self.data = input
        targets = input["targets"] # shape[-1] = K+T
        diff_in = input["diff_in"] # shape[-1] = K-1
        self.diff_in = []
        self.targets = []
        f_volatile = not self.updateG or not self.is_train
        if len(self.gpu_ids) > 0:
            for i in range(self.K - 1):
                self.diff_in.append(Variable(diff_in[:, :, :, :, i].cuda(), volatile=f_volatile))
            for i in range(self.K + self.T):
                self.targets.append(Variable(targets[:, :, :, :, i].cuda(), volatile=f_volatile))
        else:
            for i in range(self.K - 1):
                self.diff_in.append(Variable(diff_in[:, :, :, :, i], volatile=f_volatile))
            for i in range(self.K + self.T):
                self.targets.append(Variable(targets[:, :, :, :, i], volatile=f_volatile))

    def forward(self):
        # pdb.set_trace()
        self.pred = self.generator.forward(self.K, self.T, self.state, self.opt.batch_size, self.opt.image_size, self.diff_in, self.targets)

    def backward_D(self):
        # fake
        input_fake = torch.cat(self.targets[:self.K] + self.pred, dim=1)
        input_fake_ = Variable(input_fake.data)
        h_sigmoid, h = self.discriminator.forward(input_fake_, self.opt.batch_size)
        if len(self.gpu_ids) > 0:
            labels = Variable(torch.zeros(h.size()).cuda())
        else:
            labels = Variable(torch.zeros(h.size()))
        self.loss_d_fake = self.loss_d(h_sigmoid, labels)

        # real
        input_real = torch.cat(self.targets, dim=1)
        input_real_ = Variable(input_real.data)
        h_sigmoid_, h_ = self.discriminator.forward(input_real_, self.opt.batch_size)
        if len(self.gpu_ids) > 0:
            labels_ = Variable(torch.ones(h_.size()).cuda())
        else:
            labels_ = Variable(torch.ones(h_.size()))
        self.loss_d_real = self.loss_d(h_sigmoid_, labels_)

        self.loss_D = self.loss_d_fake + self.loss_d_real

        self.loss_D.backward()

    def backward_G(self):
        outputs = networks.inverse_transform(torch.cat(self.pred, dim=0))
        targets = networks.inverse_transform(torch.cat(self.targets[self.K:], dim=0))
        self.Lp = self.loss_Lp(outputs, targets)
        # pdb.set_trace()
        self.gdl = self.loss_gdl(outputs, targets)
        self.loss_G = self.opt.alpha * (self.Lp + self.gdl)

        if not self.opt.no_adversarial:
            input_fake = torch.cat(self.targets[:self.K] + self.pred, dim=1)
            h_sigmoid, h = self.discriminator.forward(input_fake, self.opt.batch_size)
            if len(self.gpu_ids) > 0:
                labels = Variable(torch.ones(h.size()).cuda())
            else:
                labels = Variable(torch.ones(h.size()))
            self.L_GAN = self.loss_d(h_sigmoid, labels)

            if not self.updateD:
                if len(self.gpu_ids) > 0:
                    labels_ = Variable(torch.zeros(h.size()).cuda())
                else:
                    labels_ = Variable(torch.zeros(h.size()))
                self.loss_d_fake = self.loss_d(h_sigmoid, labels_)

                input_real = torch.cat(self.targets, dim=1)
                input_real_ = Variable(input_real.data)
                h_sigmoid_, h_ = self.discriminator.forward(input_real_, self.opt.batch_size)
                # print('in real, h:', h_)
                if len(self.gpu_ids) > 0:
                    labels__ = Variable(torch.ones(h_.size()).cuda())
                else:
                    labels__ = Variable(torch.ones(h_.size()))
                self.loss_d_real = self.loss_d(h_sigmoid_, labels__)
            self.loss_G += self.opt.beta * self.L_GAN

        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()
        if self.opt.no_adversarial:
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        else:    
            if self.opt.D_G_switch == 'adaptive':
                if self.updateD:
                    self.optimizer_D.zero_grad()
                    self.backward_D()
                    self.optimizer_D.step()

                if self.updateG:
                    self.optimizer_G.zero_grad()
                    self.backward_G()
                    self.optimizer_G.step()

                if self.loss_d_fake.data[0] < self.opt.margin or self.loss_d_real.data[0] < self.opt.margin:
                    self.updateD = False

                if self.loss_d_fake.data[0] > (1. - self.opt.margin) or self.loss_d_real.data[0] > (1.- self.opt.margin):
                    self.updateG = False

                if not self.updateD and not self.updateG:
                    self.updateD = True
                    self.updateG = True

            elif self.opt.D_G_switch == 'alternative':
                self.optimizer_D.zero_grad()
                self.backward_D()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()
                self.backward_G()
                self.optimizer_G.step()
            else:
                raise NotImplementedError('switch method [%s] is not implemented' % self.opt.D_G_switch)


    def get_current_errors(self):
        if not self.opt.no_adversarial:
            return OrderedDict([('G_GAN', self.L_GAN.data[0]),
                                ('G_Lp', self.Lp.data[0]),
                                ('G_gdl', self.gdl.data[0]),
                                ('G_loss', self.loss_G.data[0]),
                                ('D_real', self.loss_d_real.data[0]),
                                ('D_fake', self.loss_d_fake.data[0])
                                ])
        else:
            return OrderedDict([('G_Lp', self.Lp.data[0]),
                                ('G_gdl', self.gdl.data[0]),
                                ('G_loss', self.loss_G.data[0])
                                ])

    def get_current_visuals(self):
        '''
        :return: dict, diff_in: K-1 [batch_size, h, w, c], ndarray, [0,255];
         targets: K+T ndarrays, pred: T ndarray
        '''
        if len(self.gpu_ids) > 0:
            seq_batch = self.data['targets'].cpu() # [1, 1, 128, 128, 30]
            pred = [a.data.cpu() for a in self.pred] # [1,1,128,128]
        else:
            seq_batch = self.data['targets']
            pred = [a.data for a in self.pred]
        return OrderedDict([('seq_batch', seq_batch), ('pred', pred)])

    def save(self, label, epoch):
        current_state = {
            "epoch": epoch,
            "generator": self.generator.cpu().state_dict(),
            "discriminator": self.discriminator.cpu().state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "updateD": self.updateD,
            "updateG": self.updateG,
        }
        save_filename = '%s_model.pth.tar' % (label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(current_state, save_path)
        if len(self.gpu_ids) and torch.cuda.is_available():
            self.generator.cuda(device_id=self.gpu_ids[0])
            self.discriminator.cuda(device_id=self.gpu_ids[0])

    def load(self, epoch_label):
        save_filename = '%s_model.pth.tar' % (epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.isfile(save_path):
            print("=> loading snapshot from {}".format(save_path))
            snapshot = torch.load(save_path)
            self.start_epoch = snapshot['epoch'] + 1
            self.generator.load_state_dict(snapshot['generator'])
            if self.is_train:
                self.optimizer_G.load_state_dict(snapshot["optimizer_G"])
                if not self.opt.no_adversarial:
                    self.discriminator.load_state_dict(snapshot['discriminator'])
                    self.optimizer_D.load_state_dict(snapshot["optimizer_D"])
            self.updateD = snapshot['updateD']
            self.updateG = snapshot['updateG']
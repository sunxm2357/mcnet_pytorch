import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from PIL import Image
import imageio
import numpy as np
import cv2
from util.util import *
import pdb
# DEBUG = True

class UcfDataset(BaseDataset):
    def initialize(self, opt):
        self.opt= opt
        self.root = opt.dataroot
        self.textroot = opt.textroot
        self.toTensor = transforms.ToTensor()
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        f = open(os.path.join(self.textroot, opt.video_list), 'r')
        self.files = f.readlines()
        self.K = opt.K
        self.T = opt.T
        self.debug = opt.debug
        self.backwards = opt.backwards # T/F
        self.flip = opt.flip # T/F
        self.pick_mode = opt.pick_mode
        self.image_size = opt.image_size
        self.gpu_ids = opt.gpu_ids
        if self.pick_mode in ["Slide", "First"]:
            self.backwards = False
            self.flip = False
        self.seq_len = self.K + self.T


    def __len__(self):
        if self.debug:
            return self.opt.batch_size
        else:
            return len(self.files)

    def name(self):
        return "UcfDataset"

    def read_seq(self, vid, stidx, vid_name):
        targets = []
        gray_imgs = []
        flip_flag = random.random()
        back_flag = random.random()
        for t in range(self.seq_len):
            while True:
                try:
                    img = cv2.resize(vid.get_data(stidx + t), (self.image_size[1], self.image_size[0]))[:, :, ::-1]
                    gray_img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2GRAY)
                    break
                except Exception:
                    print('in cv2', self.vid_path, stidx+t)
                    print("imageio failed loading frames, retrying")
            assert (np.max(img) > 1, "the range of image should be [0,255]")
            if len(img.shape) == 2: img = np.expand_dims(img, axis=2)
            if len(gray_img.shape) == 2: gray_img = np.expand_dims(gray_img, axis=2)
            if self.flip and flip_flag > 0.5:
                img = img[:, ::-1, :]
                gray_img = gray_img[:, ::-1, :]
            targets.append(self.toTensor(img.copy()))
            gray_imgs.append(gray_img)


        if self.backwards and back_flag > 0.5:
            targets = targets[::-1]
            gray_imgs = gray_imgs[::-1]

        diff_ins = []
        for t in range(1, self.K):
            prev = gray_imgs[t - 1] / 255.
            next = gray_imgs[t] / 255.
            diff_ins.append(torch.from_numpy(np.transpose(next - prev, axes=(2, 0, 1)).copy()).float())

        target = fore_transform(torch.stack(targets, dim=-1))
        diff_in = torch.stack(diff_ins, dim=-1)

        return {'targets': target, 'diff_in': diff_in,
                      'video_name': '%s' % (vid_name),
                      'start-end':  '%d-%d' % (stidx, stidx + self.seq_len - 1)}


    def __getitem__(self, index):
        self.files[index] = self.files[index].replace("/HandStandPushups/",
                                            "/HandstandPushups/")
        vid_name = self.files[index].split()[0]
        vid_path = os.path.join(self.root, vid_name)
        self.vid_path = vid_path
        while True:
            try:
                vid = imageio.get_reader(vid_path,"ffmpeg")
                break
            except Exception:
                print(vid_path)
                print("imageio failed loading frames, retrying")

        low = 1
        high = vid.get_length() - self.seq_len
        assert(high >= low, "the video is not qualified")
        if self.pick_mode == "Random":
            if low == high:
                stidx = low
            else:
                stidx = np.random.randint(low=low, high=high)
        elif self.pick_mode == "First":
            stidx = low
        elif self.pick_mode == "Slide":
            stidx = low
        else:
            raise NotImplementedError('pick_mode method [%s] is not implemented' % self.pick_mode)

        if not self.pick_mode == "Slide":
            input_data = self.read_seq(vid, stidx, vid_name)
        else:
            input_data = []
            n_skip = self.T
            for j in range(low, high, n_skip):
                input_data.append(self.read_seq(vid, j, vid_name))

        return input_data









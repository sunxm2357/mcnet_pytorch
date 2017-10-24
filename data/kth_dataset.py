import os.path
import random
import torchvision.transform as transforms
import torch
from data.base_dataset import BaseDataset
from PIL import Image
import imageio
import numpy as np
import cv2
from util.util import *

class KthDataset(BaseDataset):
    def initialize(self, opt):
        self.opt= opt
        self.root = opt.dataroot
        if self.is_train:
            f = open(os.path.join(self.root, 'train_data_list_trimmed.txt'), 'r')
        else:
            f = open(os.path.join(self.root, 'test_data_list.txt'), 'r')
        self.files = f.readlines()
        self.K = opt.K
        self.T = opt.T
        self.debug = opt.debug
        self.backwards = opt.backwards # T/F
        self.flip = opt.flip # T/F
        self.pick_mode = opt.pick_mode
        self.image_size = opt.image_size
        if self.debug:
            self.backwards = False
            self.filp = False
            self.pick_mode = "First"
        if self.pick_mode == "Sequential":
            self.count = np.zeros(len(self.files))


    def __len__(self):
        if self.debug:
            return self.opt.batchsize
        else:
            return len(self.trainfiles)

    def name(self):
        return "KthDataset"

    def __getitem__(self, index):
        tokens = self.files[index].split()
        vid_path = os.path.join(self.root, tokens[0]+'_uncomp.avi')
        vid = imageio.get_reader(vid_path, "ffmpeg")
        low = int(tokens[1])
        high = min([int(tokens[2]), vid.get_length()]) - self.K - self.T + 1
        assert(high >= low, "the video is not qualified")
        if self.pick_mode == "Random":
            if low == high:
                stidx = low
            else:
                stidx = np.random.randint(low=low, high=high)
        elif self.pick_mode == "First":
            stidx = low

        elif self.pick_mode == "Sequential":
            stidx = self.count[index]*(self.K + self.T) + low
            if stidx > high:
                self.count[index] = 0
                stidx = low
            self.count[index] += 1

        targets = []
        imgs = []
        flip_flag = random.random()
        back_flag = random.random()
        for t in range(self.K + self.T):
            img = cv2.cvtColor(cv2.resize(vid.get_data(stidx + t), (self.image_size, self.image_size)), cv2.COLOR_RGB2GRAY)
            assert (np.max(img) > 1, "the range of image should be [0,255]")
            if self.flip and flip_flag > 0.5:
                img = img[:, ::-1, :]
            targets.append(transforms.ToTensor(img))
            imgs.append(img)

        if self.backwards and back_flag > 0.5:
            targets = targets[::-1]
            imgs = imgs[::-1]

        diff_ins = []
        for t in range(1, self.K):
            prev = imgs[t-1]/255.
            next = imgs[t]/255.
            diff_ins.append(torch.from_numpy(np.transpose(next - prev, axes=(2, 0, 1))))

        target = fore_transform(torch.stack(targets, dim=-1))
        diff_in = torch.stack(diff_ins, dim=-1)

        return {'targets': target, 'diff_in': diff_in }









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
DEBUG = True

class KthDataset(BaseDataset):
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
        return "KthDataset"

    def read_seq(self, vid, stidx, tokens):
        targets = []
        imgs = []
        imgs_numpy = []
        flip_flag = random.random()
        back_flag = random.random()
        for t in range(self.seq_len):
            c = 0
            while True:
                try:
                    img = cv2.cvtColor(cv2.resize(vid.get_data(stidx + t), (self.image_size[1], self.image_size[0])),
                                       cv2.COLOR_RGB2GRAY)
                    break
                except Exception:
                    c = c + 1
                    if c > 5: break
                    print('in cv2', self.vid_path, stidx+t)
                    print("imageio failed loading frames, retrying")
            # if DEBUG:
            #     pdb.set_trace()
            assert (np.max(img) > 1, "the range of image should be [0,255]")
            if len(img.shape) == 2: img = np.expand_dims(img, axis=2)
            if self.flip and flip_flag > 0.5:
                img = img[:, ::-1, :]
            # pdb.set_trace()
            targets.append(self.toTensor(img.copy()))
            imgs.append(img)


        if self.backwards and back_flag > 0.5:
            targets = targets[::-1]
            imgs = imgs[::-1]
            imgs_numpy = imgs_numpy[::-1]

        diff_ins = []
        for t in range(1, self.K):
            prev = imgs[t - 1] / 255.
            next = imgs[t] / 255.
            diff_ins.append(torch.from_numpy(np.transpose(next - prev, axes=(2, 0, 1)).copy()).float())

        target = fore_transform(torch.stack(targets, dim=-1))
        diff_in = torch.stack(diff_ins, dim=-1)

        return {'targets': target, 'diff_in': diff_in,
                      'video_name': '%s_%s_%s' % (tokens[0], tokens[1], tokens[2]),
                      'start-end':  '%d-%d' % (stidx, stidx + self.seq_len - 1)}


    def __getitem__(self, index):
        tokens = self.files[index].split()
        # with open(self.log_name, 'a') as log_file:
        #     log_file.write(tokens[0])
        # print(tokens[0])
        vid_path = os.path.join(self.root, tokens[0]+'_uncomp.avi')
        self.vid_path = vid_path
        while True:
            try:
                vid = imageio.get_reader(vid_path,"ffmpeg")
                break
            except Exception:
                print(vid_path)
                print("imageio failed loading frames, retrying")
        low = int(tokens[1])
        high = min([int(tokens[2]), vid.get_length()]) - self.seq_len + 1
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
            input_data = self.read_seq(vid, stidx, tokens)
        else:
            input_data = []
            action = vid_path.split("_")[1]
            if action in ["running", "jogging"]:
                n_skip = 3
            else:
                n_skip = self.T
            for j in range(low, high, n_skip):
                input_data.append(self.read_seq(vid, j, tokens))

        return input_data









from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    """
    :param image_tensor: [batch_size, c, h, w]
    :param imtype: np.uint8
    :return: ndarray [batch_size, c, h, w]
    """
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[1] == 1:
        image_numpy = np.tile(image_numpy, (1, 3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def tensorlist2imlist(tensors):
    ims = []
    for tensor in tensors:
        ims.append(tensor2im(tensor.data))
    return ims

def inverse_transform(images):
    return (images+1.)/2


def fore_transform(images):
    return images * 2 - 1


def bgr2gray(image):
    # rgb -> grayscale 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray_ = 0.1140 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.2989 * image[:, 2, :, :]
    gray = torch.unsqueeze(gray_, 1)
    return gray


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def draw_frame(img, is_input):
    if img.shape[2] == 1:
        img = np.repeat(img, [3], axis=2)

    if is_input:
        img[:2,:,0]  = img[:2,:,2] = 0
        img[:,:2,0]  = img[:,:2,2] = 0
        img[-2:,:,0] = img[-2:,:,2] = 0
        img[:,-2:,0] = img[:,-2:,2] = 0
        img[:2,:,1]  = 255
        img[:,:2,1]  = 255
        img[-2:,:,1] = 255
        img[:,-2:,1] = 255
    else:
        img[:2,:,0]  = img[:2,:,1] = 0
        img[:,:2,0]  = img[:,:2,1] = 0
        img[-2:,:,0] = img[-2:,:,1] = 0
        img[:,-2:,0] = img[:,-2:,1] = 0
        img[:2,:,2]  = 255
        img[:,:2,2]  = 255
        img[-2:,:,2] = 255
        img[:,-2:,2] = 255
    return img


def draw_err_plot(err, path, err_name):
    avg_err = np.mean(err, axis=0)
    T = err.shape[1]
    # plt.clf()
    ax = plt.figure().gca()
    x = np.arange(1, T+1)
    plt.plot(x, avg_err, marker="d")
    plt.xlabel('time steps')
    plt.ylabel(err_name)
    plt.grid()
    ax.set_xticks(x)
    plt.savefig(path)

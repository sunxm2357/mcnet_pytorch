from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def tensorlist2imlist(tensors):
    ims = []
    for tensor in tensors:
        ims.append(tensor2im(tensor))
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
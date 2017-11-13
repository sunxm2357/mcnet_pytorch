from __future__ import print_function
import torch
import io
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
import torchvision.utils as vutils



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

def draw_frame_tensor(img, K, T):
    img[:K, 0, :2, :] = img[:K, 2, :2, :] = 0
    img[:K, 0, :, :2] = img[:K, 2, :, :2] = 0
    img[:K, 0, -2:, :] = img[:K, 2, -2:, :] = 0
    img[:K, 0, :, -2:] = img[:K, 2, :, -2:] = 0
    img[:K, 1, :2, :] = 1
    img[:K, 1, :, :2] = 1
    img[:K, 1, -2:, :] = 1
    img[:K, 1, :, -2:] = 1
    img[K:K+T, 0, :2, :] = img[K:K+T, 1, :2, :] = 0
    img[K:K+T, 0, :, :2] = img[K:K+T, 1, :, :2] = 0
    img[K:K+T, 0, -2:, :] = img[K:K+T, 1, -2:, :] = 0
    img[K:K+T, 0, :, -2:] = img[K:K+T, 1, :, -2:] = 0
    img[K:K+T, 2, :2, :] = 1
    img[K:K+T, 2, :, :2] = 1
    img[K:K+T, 2, -2:, :] = 1
    img[K:K+T, 2, :, -2:] = 1
    return img


def draw_err_plot(err,  err_name, lims, path=None, type="Val"):
    avg_err = np.mean(err, axis=0)
    T = err.shape[1]
    fig = plt.figure()
    # plt.clf()
    ax = fig.add_subplot(111)
    x = np.arange(1, T+1)
    ax.plot(x, avg_err, marker="d")
    ax.set_xlabel('time steps')
    ax.set_ylabel(err_name)
    ax.grid()
    ax.set_xticks(x)
    ax.axis(lims)
    if type == 'Val':
        plot_buf = gen_plot(fig)
        im = np.array(Image.open(plot_buf), dtype=np.uint8)
        plt.close(fig)
        return im
    elif type == 'Test':
        plt.savefig(path)
    else:
        raise ValueError('error plot type [%s] is not defined' % type)

def plot_to_image(x, y, lims):
    '''
    Plot y vs. x and return the graph as a NumPy array
    :param x: X values
    :param y: Y values
    :param lims: [x_start, x_end, y_start, y_end]
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.axis(lims)
    plot_buf = gen_plot(fig)
    im = np.array(Image.open(plot_buf), dtype=np.uint8)
    im = np.expand_dims(im, axis=0)
    plt.close(fig)
    return im


def gen_plot(fig):
    """
    Create a pyplot plot and save to buffer.
    https://stackoverflow.com/a/38676842
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf


def visual_grid(seq_batch, pred, K, T):
    pred_data = torch.stack(pred, dim=-1)

    true_data = seq_batch[:, :, :, :, K:K + T].clone()

    pred_data = torch.cat([seq_batch[:, :, :, :, :K], pred_data], dim=-1)
    true_data = torch.cat((seq_batch[:, :, :, :, :K], true_data), dim=-1)
    batch_size = int(pred_data.size()[0])
    c_dim = int(pred_data.size()[1])
    vis = []
    for i in range(batch_size):
        # pdb.set_trace()
        pred_data_sample = inverse_transform(pred_data[i, :, :, :, :]).permute(3, 0, 1, 2)
        target_sample = inverse_transform(true_data[i, :, :, :, :]).permute(3, 0, 1, 2)
        if c_dim == 1:
            pred_data_sample = torch.cat([pred_data_sample]*3, dim=1)
            target_sample = torch.cat([target_sample]*3, dim=1)
        pred_data_sample = draw_frame_tensor(pred_data_sample, K, T)
        target_sample = draw_frame_tensor(target_sample, K, T)
        output = torch.cat([pred_data_sample, target_sample], dim=0)
        vis.append(vutils.make_grid(output, nrow=K+T))
    grid = torch.cat(vis, dim=1)
    # pdb.set_trace()
    grid = torch.from_numpy(np.flip(grid.numpy(), 0).copy())
    return grid
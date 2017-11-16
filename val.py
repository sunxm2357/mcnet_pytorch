from options.test_options import TestOptions
from models.models import create_model
from data.data_loader import *
from util.util import *
from os import system
from skimage.measure import compare_ssim as ssim
import skimage.measure as measure
import cv2
import os
import torchvision.utils as vutils
import copy
import pdb
from PIL import Image
# from PIL import ImageDraw


def val(opt):
    multichannel = not (opt.c_dim == 1)
    if opt.data == "KTH":
        lims_ssim = [1, opt.T, 0, 1]
        lims_psnr = [1, opt.T, 0, 34]
    elif opt.data == 'UCF':
        lims_ssim = [1, opt.T, 0, 1]
        lims_psnr = [1, opt.T, 0, 35]
    else:
        raise ValueError('Dataset [%s] not recognized.' % opt.data)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# val videos = %d' % dataset_size)

    model = create_model(opt)

    psnr_err = np.zeros((0, opt.T))
    ssim_err = np.zeros((0, opt.T))

    for i, datas in enumerate(dataset):
        if opt.pick_mode == 'First': datas = [datas]
        for data in datas:
            model.set_inputs(data)
            model.forward()

            if len(opt.gpu_ids) > 0:
                seq_batch = data['targets'].cpu().numpy().transpose(0, 2, 3, 4, 1)
                pred = [a.data.cpu().numpy().transpose((0, 2, 3, 1)) for a in model.pred]
            else:
                seq_batch = data['targets'].numpy().transpose(0, 2, 3, 4, 1)
                pred = [a.data.numpy().transpose((0, 2, 3, 1)) for a in model.pred]

            pred_data = np.stack(pred, axis=3)
            true_data = seq_batch[:, :, :, opt.K:opt.K + opt.T, :].copy()

            cpsnr = np.zeros((opt.T,))
            cssim = np.zeros((opt.T,))
            for t in range(opt.T):
                pred = (inverse_transform(pred_data[0, :, :, t]) * 255).astype("uint8")
                target = (inverse_transform(true_data[0, :, :, t]) * 255).astype("uint8")
                if opt.c_dim == 1:
                    pred = np.squeeze(pred, axis=-1)
                    target = np.squeeze(target, axis=-1)
                cpsnr[t] = measure.compare_psnr(pred, target)
                cssim[t] = ssim(target, pred, multichannel=multichannel)
            # pdb.set_trace()
            psnr_err = np.concatenate((psnr_err, cpsnr[None, :]), axis=0)
            ssim_err = np.concatenate((ssim_err, cssim[None, :]), axis=0)

    psnr_plot = draw_err_plot(psnr_err, 'Peak Signal to Noise Ratio', lims=lims_psnr)
    ssim_plot = draw_err_plot(ssim_err, 'Structural Similarity', lims=lims_ssim)

    vis_opt = copy.deepcopy(opt)
    vis_opt.video_list = 'vis_data_list.txt'
    vis_opt.pick_mode = 'First'
    vis_data_loader = CreateDataLoader(vis_opt)
    vis_dataset = vis_data_loader.load_data()
    vis_dataset_size = len(vis_data_loader)
    print('# visualization videos = %d' % vis_dataset_size)
    vis = []
    for i, data in enumerate(vis_dataset):
        model.set_inputs(data)
        model.forward()
        visuals = model.get_current_visuals()
        vis.append(visual_grid(visuals['seq_batch'], visuals['pred'], opt.K, opt.T))
    grid = torch.cat(vis, dim=1)
    print("Validation done.")
    return psnr_plot, ssim_plot, grid

from options.test_options import TestOptions
from models.models import create_model
from data.data_loader import *
from util.visualizer import Visualizer
from util.util import *
import sys
from os import system
# import ssim
from skimage.measure import compare_ssim as ssim
import skimage.measure as measure
import cv2
import os
import pdb
from PIL import Image
# from PIL import ImageDraw

def main():
    opt = TestOptions().parse()
    if opt.data == "KTH":
        lims_ssim = [1, opt.T, 0.6, 1]
        lims_psnr = [1, opt.T, 20, 34]
    else:
        raise ValueError('Dataset [%s] not recognized.' % opt.data)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# testing videos = %d' % dataset_size)

    model = create_model(opt)

    psnr_err = np.zeros((0, opt.T))
    ssim_err = np.zeros((0, opt.T))

    for i, datas in enumerate(dataset):
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

            true_data = seq_batch[:,:,:,opt.K:opt.K + opt.T,:].copy()

            pred_data = np.concatenate((seq_batch[:, :, :, :opt.K, :], pred_data), axis=3)
            true_data = np.concatenate((seq_batch[:, :, :, :opt.K, :], true_data), axis=3)

            cpsnr = np.zeros((opt.K + opt.T,))
            cssim = np.zeros((opt.K + opt.T,))
            for t in range(opt.K + opt.T):
                pred = (inverse_transform(pred_data[0, :, :, t]) * 255).astype("uint8")
                target = (inverse_transform(true_data[0, :, :, t]) * 255).astype("uint8")
                if opt.c_dim == 1:
                    pred = np.squeeze(pred, axis=-1)
                    target = np.squeeze(target, axis=-1)
                cpsnr[t] = measure.compare_psnr(pred, target)
                # pdb.set_trace()
                cssim[t] = ssim(target, pred)
                # cssim[t] = ssim(Image.fromarray(target),Image.fromarray(pred))
                if opt.c_dim == 1:
                    pred = np.expand_dims(pred, axis=-1)
                    target = np.expand_dims(target, axis=-1)

                pred = draw_frame(pred, t < opt.K)
                target = draw_frame(target, t < opt.K)

                savedir = os.path.join(opt.save_dir, data['video_name'][0].split('.')[0] + '_' + data['start-end'][0])
                makedir(savedir)

                cv2.imwrite(savedir + "/pred_" + "{0:04d}".format(t) + ".png", pred)
                cv2.imwrite(savedir + "/gt_" + "{0:04d}".format(t) + ".png", target)

            cmd1 = "rm " + savedir + "/pred.gif"
            cmd2 = ("ffmpeg -f image2 -framerate 3 -i " + savedir +
                    "/pred_%04d.png " + savedir + "/pred.gif")
            cmd3 = "rm " + savedir + "/pred*.png"

            # Comment out "system(cmd3)" if you want to keep the output images
            # Otherwise only the gifs will be kept
            system(cmd1); system(cmd2)
            system(cmd3)

            cmd1 = "rm " + savedir + "/gt.gif"
            cmd2 = ("ffmpeg -f image2 -framerate 3 -i " + savedir +
                    "/gt_%04d.png " + savedir + "/gt.gif")
            cmd3 = "rm " + savedir + "/gt*.png"

            # Comment out "system(cmd3)" if you want to keep the output images
            # Otherwise only the gifs will be kept
            system(cmd1); system(cmd2)
            system(cmd3)

            psnr_err = np.concatenate((psnr_err, cpsnr[None, opt.K:]), axis=0)
            ssim_err = np.concatenate((ssim_err, cssim[None, opt.K:]), axis=0)

    save_path = os.path.join(opt.quant_dir, 'results_model=' + opt.name + '_' + opt.which_epoch + '.npz')
    np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
    psnr_plot = os.path.join(opt.quant_dir, 'results_model=' + opt.name + '_' + opt.which_epoch + 'psnr.png')
    draw_err_plot(psnr_err, 'Peak Signal to Noise Ratio', path=psnr_plot, lims=lims_psnr, type="Test")
    ssim_plot = os.path.join(opt.quant_dir, 'results_model=' + opt.name + '_' + opt.which_epoch + 'ssim.png')
    draw_err_plot(psnr_err, 'Structural Similarity', path=ssim_plot, lims=lims_ssim, type="Test")
    print("Results saved to " + save_path)
    print("Done.")

if __name__ == "__main__":
    main()
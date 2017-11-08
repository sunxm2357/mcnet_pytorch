import numpy as np
import argparse
import os
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help="the path of the .npz file")
parser.add_argument('--psnr_plot', type=str, required=True, help="the path to save psnr.png")
parser.add_argument('--ssim_plot', type=str, required=True, help="the path to save ssim.png")
opt = parser.parse_args()
metrics = np.load(opt.path)
draw_err_plot(metrics['psnr'], opt.psnr_plot, 'Peak Signal to Noise Ratio')
draw_err_plot(metrics['ssim'], opt.ssim_plot, 'Structural Similarity')
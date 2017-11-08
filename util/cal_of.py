import cv2
import pdb
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.stats import entropy
from numpy.linalg import norm


def flow(path1, path2):
    img1 = cv2.cvtColor(cv2.imread(path1),cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread(path2),cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, required=True, help="the path containing all the test data")
parser.add_argument('--K', type=int, required=True, help="the number of inputs")
parser.add_argument('--T', type=int, required=True, help="the number of outputs")
parser.add_argument('--output_path', type=str, required=True, help='path to save the output')
opt = parser.parse_args()
flow_ins = []
flow_outs = []
for i, dir in enumerate(os.listdir(opt.test_path)):
    dir_name = os.path.join(opt.test_path, dir)
    if os.path.isdir(dir_name):
        print(i, dir)
        path_i_1 = os.path.join(dir_name, 'pred_%04d.png' % (opt.K-2))
        path_i_2 = os.path.join(dir_name, 'pred_%04d.png' % (opt.K-1))
        path_o_1 = os.path.join(dir_name, 'pred_%04d.png' % (opt.K))
        path_o_2 = os.path.join(dir_name, 'pred_%04d.png' % (opt.K+1))
        flow_in = flow(path_i_1, path_i_2)
        flow_out = flow(path_o_1, path_o_2)
        flow_ins.append(flow_in.std())
        flow_outs.append(flow_out.std())


flow_stds = [flow_ins, flow_outs]
npy_path = os.path.join(opt.output_path, 'flow_std.npy')
np.save(npy_path, np.array(flow_stds))
range_max = max([max(flow_ins), max(flow_outs)])
range_min = min([min(flow_ins), min(flow_outs)])
plt.clf()
plt.hist(flow_ins, bins=100, range=(range_min, range_max))
plt.xlabel('std for optical flows')
plt.title('optical flow for the last two inputs')
plt.savefig(os.path.join(opt.output_path, 'optical_flow_input.png'))
plt.clf()
plt.hist(flow_outs, bins=100, range=(range_min, range_max))
plt.xlabel('std for optical flows')
plt.title('optical flow for the first two outputs')
plt.savefig(os.path.join(opt.output_path, 'optical_flow_output.png'))
print('average std for input flow', sum(flow_ins)/float(len(flow_ins)))
print('average std for output flow', sum(flow_outs)/float(len(flow_outs)))
hist1, _ = np.histogram(flow_ins, bins=100, range=(range_min, range_max))
# hist1_prob = hist1/float(np.sum(hist1))
hist2, _ = np.histogram(flow_outs, bins=100, range=(range_min, range_max))
# hist2_prob = hist2/float(np.sum(hist2))
jsd = JSD(hist1, hist2)
print('Jensen-Shannon divergence', jsd)

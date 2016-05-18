from __future__ import division
__author__ = 'imamanoor'
# if you use another test/train set change number of classes and the
# unlabeled index as well as number of iterations (needs to be equal to the test set size)
import cv2
import numpy as np
import os
import caffe
import numpy as np
import os
from datetime import datetime
from PIL import Image
import argparse

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, iter, dataset, layer='prob', gt='label'):
    n_cl = net.blobs[layer].channels
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in range(0,iter):
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)
    return hist / iter

def seg_tests(net, iter, dataset, layer='prob', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    do_seg_tests(net, iter, dataset, layer, gt)

def do_seg_tests(net, iter, dataset, layer='prob', gt='label'):
    n_cl = net.blobs[layer].channels

    hist, loss = compute_hist(net, iter, dataset, layer, gt)
    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    return hist

DVT_ROOT = '/Users/imamanoor/Documents/jobApplications/noor-vision-deep-learning-segmentation/vision-deep-learning-segmentation/'
gtPath = DVT_ROOT + '/forCNN/gt/' # path to your ground truth images
predPath = DVT_ROOT + '/forCNN/predictions/' #path to your predictions (you get them after you implement saving images in the test_segmentation_camvid.py script - or you write your own)
groundTruths = os.listdir(gtPath)
skip = 0 # first two are '.' and '..' so skip them
predictions = os.listdir(predPath)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_cpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)
numClasses = 2
count = 400

seg_tests(net, count, net.blobs['data'], layer='prob', gt='label')

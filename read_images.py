import os
import numpy as np
from PIL import Image
from PIL import ImageOps
import logging
import caffe
import cv2

logging.basicConfig(level = logging.DEBUG)
logging.debug("BEGIN")

X = None            ## placeholder for input samples
y = None            ## placeholder for output label masks
NumSample = 2000    ## the number of samples to read from the directory
PIXELS_IN_X = 121   #480 downsample the input samples to this resolution (PIXELS_IN x PIXELS_IN)
PIXELS_IN_Y = 121   #360
PIXELS_OUT_X = 121  #480 downsample the output labels mask to this resolution - useful when you're pooling and need the output to be of a smaller resolution
PIXELS_OUT_Y = 121   #360
MODEL_FILE = 'seg_solver.prototxt'
PRETRAINED = 'VGG_ILSVRC_16_layers.caffemodel'
abs_PATH = '/noor-vision-deep-learning-segmentation/vision-deep-learning-segmentation/'

def ReadImages():
    global NumSample, X, y
    
    d = "./forCNN"
    logging.debug("Read directory: " + d)
    im = cv2.imread('mean.png')
    filenames = [os.path.join(d, f) for f in os.listdir(d)]
    
    if NumSample is None:
        NumSample = len(filenames)
        
    X = np.zeros((NumSample, 1, PIXELS_IN_Y, PIXELS_IN_X), dtype='float32')
    y = np.zeros((NumSample, PIXELS_OUT_Y * PIXELS_OUT_X), dtype='float32')
    file = open("train_test_file.txt", "w")
    for i in range(0, NumSample):
        filefullpath = filenames[i]
        filename = os.path.basename(filefullpath)
            
        fileid = None
        if filename[0:5] == "depth":
            fileid = filename[5:10]
        else:
            continue

        img = Image.open(filefullpath).convert('L')
        img = ImageOps.fit(img, (PIXELS_IN_X, PIXELS_IN_Y), Image.ANTIALIAS)
        img = np.asarray(img, dtype='float32')
        X[i] = img
        
        labels = Image.open(os.path.join(d, "mask" + fileid + ".png")).convert('L')
        labels = ImageOps.fit(labels, (PIXELS_OUT_X, PIXELS_OUT_Y), Image.NEAREST)
        labels = np.asarray(labels, dtype = 'float32') / 50
        labels = labels
        idx = labels > 0
        labels[idx] = 1
        y[i] = labels.reshape(1, PIXELS_OUT_X * PIXELS_OUT_Y)
        cv2.imwrite('forCNN/resize_121/mask'+fileid+'.jpg',labels)
        cv2.imwrite('forCNN/resize_121/depth'+fileid+'.png',img)
        path_str = abs_PATH + 'forCNN/resize_3/depth'+fileid+'.png '+ abs_PATH + 'forCNN/resize_3/mask'+fileid+'.png '
        file.write(path_str)

def train():

    caffe.set_mode_cpu()
    solver = caffe.SGDSolver(MODEL_FILE)
    solver.net.copy_from('VGG_ILSVRC_16_layers.caffemodel')
    solver.Solve()

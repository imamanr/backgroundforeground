__author__ = 'imamanoor'
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import caffe


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--imagepath', type=str,required=True)
args = parser.parse_args()

caffe.set_mode_cpu()

net = caffe.Net(args.model,
              args.weights,
              caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]

image = cv2.imread(args.imagepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
image = cv2.resize(image,(121,121))
im = np.zeros((1,121,121))
im[0,:,:] = image
im = np.transpose(im,(2,1,0))
net.blobs['data'].data[...] = transformer.preprocess('data', im)

# perform classification
net.forward()

# obtain the output probabilities
output_prob = net.blobs['prob'].data[0]

background = [0]
foreground = [1]
predicted = net.blobs['prob'].data
labels = np.array([background, foreground])
output = np.squeeze(predicted[0,:,:,:])
ind = np.argmax(output, axis=0)
r = ind.copy()
for l in range(1):
    r[ind==l] = labels[l, 0]

plt.figure()
plt.imshow(r,vmin=0, vmax=1)
plt.hold()
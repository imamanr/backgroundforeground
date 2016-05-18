import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import caffe
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.imshow(data); plt.axis('off')
# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_cpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)


for i in range(0, args.iter):
    net.forward()
    image = net.blobs['data'].data
    label = net.blobs['label'].data
    predicted = net.blobs['prob'].data
    unique, counts = np.unique(predicted, return_counts=True)

    image = np.squeeze(image[0,0,:,:])
    output = np.squeeze(predicted[0,:,:,:])
    ind = np.argmax(output, axis=0)
    r = ind.copy()
    r_gt = label[0,0,:,:].copy()

    background = [0]
    foreground = [1]

    label_colours = np.array([background, foreground])
    for l in range(1):
        r[ind==l] = label_colours[l, 0]
        r_gt[label[0,0,:,:]==l] = label_colours[l, 0]

    rgb = np.zeros((ind.shape[0], ind.shape[1]))
    rgb[:,:] = r
    rgb_gt = np.zeros((ind.shape[0], ind.shape[1]))
    rgb_gt[:,:] = r_gt

    image = image/255.0

    filters = net.params['conv1_1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1))
    feat = net.blobs['conv1_1'].data[0, :36]
    vis_square(feat)

    feat = net.blobs['conv5_1'].data[0]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(feat.flat)
    plt.subplot(2, 1, 2)
    _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
    feat = net.blobs['conv5_1'].data[0, :100]
    vis_square(feat)


    feat = net.blobs['prob'].data[0]
    plt.figure(figsize=(15, 3))
    plt.plot(feat.flat)
    plt.figure()
    plt.imshow(image,vmin=0, vmax=1)
    plt.figure()
    plt.imshow(rgb_gt,vmin=0, vmax=1)
    plt.figure()
    plt.imshow(rgb,vmin=0, vmax=1)
    plt.show()

    plt.close('all')
print 'Success!'



Instruction to run script 
-------------------------

To run the demo you will need segnet modified caffe branch. please install it from:
https://github.com/alexgkendall/caffe-segnet

train.txt has training image and labels, test.txt has test images and labels please replace relative 
path(/noor-vision-deep-learning-segmentation/vision-deep-learning-segmentation/forCNN/resize_121)
 with absolute path on your machine.
 
please use deploy_seg.prototxt as deployment architecture file, give image path in predict
please use SEG_test.prototxt for testing, update the test.txt paths and data path on prototxt
please use train_val.prototxt for training architecture file, update the test.txt paths and data path on prototxt
seg_iter_full.caffemodel are the weights 
specify an image path in predict.py to run the model
weights visualizations are provided for first and sixth conv layer using the visualize.py
test results can be reproduced using test_results.py

Net Architecture
--------------------

Below are the layers used to learn from training samples:

data	(4, 3, 121, 121)
label	(4, 1, 121, 121)
conv1_1	(4, 64, 121, 121)
conv1_2	(4, 64, 121, 121)
pool1	(4, 64, 61, 61)
conv2_1	(4, 128, 61, 61)
conv2_2	(4, 128, 61, 61)
pool2	(4, 128, 31, 31)
conv3_1	(4, 256, 31, 31)
conv3_2	(4, 256, 31, 31)
conv3_3	(4, 256, 31, 31)
pool3	(4, 256, 16, 16)
conv4_1	(4, 512, 16, 16)
conv4_2	(4, 512, 16, 16)
conv4_3	(4, 512, 16, 16)
pool4	(4, 512, 8, 8)
conv5_1	(4, 4096, 8, 8)
conv6	(4, 2, 8, 8)
upscore2	(4, 2, 121, 121)
prob	(4, 2, 121, 121)

Evaluation Metrics
--------------------
These metrics are calculated at iteration: 36k, 
 overall accuracy 0.8070533092
 mean accuracy 0.531250976953
 mean IU 0.434697537607
 fwavacc 0.758636111692
 
 
Visulaizations of layers 
------------------------

observations:

The filters in first layers are not very sharp indicating the network is not trained fully.
The prediction layer values are mirroring in both classes indicating correct class distribution.
The last convolutional layer filters are displaying random activations


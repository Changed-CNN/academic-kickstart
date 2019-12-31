+++

title= "Semantic segmentation in PASCAL VOC based on FCN optimized by SyncBN"
date= 2019-09-01
tags=["DeepLearning", "CV", "FCN", "MXnet", "Segmentation", "SyncBN"]

+++

## Project Introduction

Standard semantic segmentation is also called `full-pixel semantic segmentation`, it classifies each pixel into classes belonging to different objects and identifies the content and location of the image by looking up all the pixels of each class. Here we will use the `MXnet` deep learning framework developed by `Amazon` to implement  semantic segmentation，and `Fully Connected Network`(FCN) (Proposed in `CVPR2015`, it is currently the most widely used network in the field of semantic segmentation) will be used to realize semantic segmentation in `PASCAL VOC `dataset. `Mean Intersection over Union`(mIoU) is our standard to evaluation the result, and `Cross-GPU Synchronized Batch Normalization`(SyncBN) will be applied to improve the effect. This task was completed in `the key laboratory of speech and image processing` in late 2018.

## Outline

- [Dataset](#1)
  - [Get The Dataset](#2)
  - [Image Transform](#3)
  - [Data Augmentation](#4)
- [Network Model](#5)
- [Cross-GPU SyncBN](#6)
- [Evaluation Standard](#7)
- [Training Details](#8)
  - [Loss](#9)
  - [Learning Rate](#10)
  - [Dataparallel](#11)
  - [Optimizer](#12)
  - [Process](#13)
- [Visualization](#14)
- [Experiment Condition](#15)

## Experiment Contents

### <span id="1">Dataset</span>

#### <span id="2">Get The Dataset</span>

We can get the `PASCAL VOC` dataset from `gluoncv.data` because semantic segmentation datasets are provided by it.

```python
from gluoncv import data
trainset = gluoncv.data.VOCSegmentation(split='train', transform=input_transform)
testset = data.VOCSegmentation(split='test', transform=input_transform)
# Create Training Loader and Test Loader
train_data = gluon.data.DataLoader(
    trainset, batch_size=16, shuffle=True, last_batch='rollover',
    num_workers=batch_size)
test_data = gluon.data.DataLoader(
    testset, batch_size=16, shuffle=True, last_batch='rollover',
    num_workers=batch_size)
```

#### <span id="3">Image Transform </span>

We need to unify the image transformation by normalizing the color of the image，and it also provided by `gluoncv.data`

```python
from mxnet.gluon.data.vision import transforms
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])
```

#### <span id="4">Data Augmentation</span>

For data augmentation, we follow the standard data augmentation routine to transform the input image and the ground truth label map synchronously. We `rotate` the image between (-20,20) degrees randomly, and `crop` the image with padding if needed. Finally a random `Gaussian blurring` is applied. You can refer to the data-enhancement part we mentioned in [Object detection and key point location based on ResNet](http://fengzhikang.xyz/2019/08/28/Object-detection-and-key-point-location-based-on-ResNet.html).

### <span id="5">Network Model</span>

**What Is FCN?**

Generally, the last few layers of the network used by the classification task are the full connection layer, which crushes the original two-dimensional matrix into one dimension, thus losing the spatial information. Finally, the training outputs a scalar, which is the classification label. The output of semantic segmentation needs to be a segmentation graph, regardless of size, but at least two-dimensional. Therefore, we need to discard the full connection layer, replace it with the full convolution layer, and extend the classification of image level to the classification of pixel level.

![ss8](/img/ss/ss8.png)

FCN replaces the full connection layer behind the traditional convolutional network with the convolutional layer, so that the network output is no longer a category but a heatmap. Meanwhile, in order to solve the influence of convolution and pooling on image size, the method of up-sampling is proposed. The up-sampling is simply the inverse process of pooling. The data quantity decreases after pooling and it increases after up-sampling. It can be understood that the up-sample is to enlarge the feature graph which is much smaller than the original image to the size of the original image.

![ss10](/img/ss/ss10.png)

Skip structure combines results of different depth layers while ensuring robustness and accuracy. The function of this structure is to optimize result. Because the result obtained by direct up-sampling after full convolutional layer is very rough, the results of different pooling layers are combined to optimize the output after up-sampling.

![ss11](/img/ss/ss11.png)

For more details and specific definition you can review [Here](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

**How To Get It?**

FCN model is provided in `gluoncv.model_zoo.FCN`. To get FCN model using ResNet50 base network for Pascal VOC dataset:

```python
model=gluoncv.model_zoo.get_fcn(dataset='pascal_voc',backbone='resnet50',pretrained=False)
```

### <span id="6">Cross-GPU SyncBN</span>

**Why We Need It?**

Standard BN implementation using Data Parallel only normalize the data within each device which is equivalent to reducing the batch-size. For training tasks that consume a lot of GPU, the batch-size on a single GPU is often too small, which affects the convergence effect of the model. Cross-GPU synchronization Batch Normalization can normalize global samples, equivalent to increasing the batch-size, so the training effect is no longer affected by the number of GPU we used. In the paper of image segmentation and object detection, the use of cross-GPU SyncBN will also significantly improve the experimental effect.

![ss1](/img/ss/ss1.png)

 SyncBN normalizes the input within the whole mini-batch. The key of SyncBN is to obtain the `global mean` and `global variance` in the forward operation and the corresponding `global gradient` in the backward operation. The easiest way is to work out the `mean` synchronously, then send it back to each GPU and work out the `variance` synchronously, but this way synchronizes twice. In fact, we only need to synchronize once. In the forward operation, we only need to calculate `AND` on each GPU, then calculate the `global sum` by Cross-GPU to get the correct `mean` and `variance`. Similarly, we only need to synchronize once in the backward operation to work out the corresponding `gradient AND`. 

![ss2](/img/ss/ss2.png)

With SyncBN, we don't need to worry about using multi-GPU to affect the convergence effect, because no matter how many GPU are used, the same global batch -size will get the same effect. For more details you can refer to [Context Encoding for Semantic Segmentation](https://arxiv.org/pdf/1803.08904.pdf) 

![ss3](/img/ss/ss3.png)

And you can get more information from [SyncBN on MXnet Gluon](https://zh.mxnet.io/blog/syncbn).

**How To Realize It?**

 MXnet Gluon provides an API to implement such a batch normalization layer:

```python
class mxnet.gluon.contrib.nn.SyncBatchNorm(in_channels=0, num_devices=None, momentum=0.9, epsilon=1e-05, center=True, scale=True, use_global_stats=False, beta_initializer='zeros', gamma_initializer='ones', running_mean_initializer='zeros', running_variance_initializer='ones', **kwargs)
```

The SyncBN part in our code:

```python
if num_sync_bn_devices == -1:#We have 2 GPUs,so num_sync_bn_devices is 2.
 self.block.add(nn.BatchNorm())
else:            self.block.add(gluon.contrib.nn.SyncBatchNorm(num_devices=num_sync_bn_devices))
```

### <span id="7">Evaluation Standard</span>

There are many standards to measure the Accuracy of the algorithm in image segmentation, which are usually variations of `Pixel Accuracy`(PA) and `Intersection over Union`(IoU), like `Mean Pixel Accuracy`(MPA), ` Mean Intersection over Union`(MIoU), `Frequency Weighted Intersection over Union`(FWIoU). We selected `MIoU` as our standard to measure our semantic segmentation task.

**What Is IoU?**

As a visual example, let's suppose we're tasked with calculating the IoU score of the following prediction, given the ground truth labeled mask.

![ss4](/img/ss/ss4.png)

In fact, the image is divided into four parts: 

1. true negative (TN, it's background and predicted as background)

2. false negative (FN, it's label but predicted as background)

3. false positive (FP, it's background but predicted as label)

4. true positive (TP, it's label and predicted as label)

**IoU=TP/(FN+FP+TP)**

![ss6](/img/ss/ss6.png)

![ss5](/img/ss/ss5.png)

The IoU is usually calculated on the basis of classes, sometimes also on the basis of images. The IoU score is calculated for each class separately and then averaged over all classes to provide a global, mean IoU score of our semantic segmentation prediction, So what we obtain at final is `MIoU`.
The code to realize it:

```python
intersection = np.logical_and(target, prediction)
union = np.logical_or(target, prediction)
iou_score = np.sum(intersection) / np.sum(union)
```

For more relative information ,you can refer to [Jeremy's Blog](https://www.jeremyjordan.me/evaluating-image-segmentation-models/).

### <span id="8">Training Details</span>

#### <span id="9">Loss</span>

We apply a standard per-pixel Softmax Cross Entropy Loss to train FCN

```python
from gluoncv.loss import MixSoftmaxCrossEntropyLoss
criterion = MixSoftmaxCrossEntropyLoss(aux=True)
```

#### <span id="10">Learning Rate</span>

We use a poly-like learning rate scheduler for FCN training, and**` lr=base_lr×(1−iter)^power`**

```python
lr_scheduler = gluoncv.utils.LRScheduler('poly', base_lr=0.001,
                                         nepochs=50, iters_per_epoch=len(train_data), 											 power=0.9)
```

#### <span id="11">Dataparallel</span>

We believe multi-GPU training can accelerate the process

```python
from gluoncv.utils.parallel import *
ctx_list = [mx.cpu(0)]
model = DataParallelModel(model, ctx_list)
criterion = DataParallelCriterion(criterion, ctx_list)
```

#### <span id="12">Optimizer</span>

We select SGD solver as the kernel of this optimizer

```python
kv = mx.kv.create('device')
optimizer = gluon.Trainer(model.module.collect_params(), 'sgd',
                          {'lr_scheduler': lr_scheduler,
                           'wd':0.0001,
                           'momentum': 0.9,
                           'multi_precision': True},
                          kvstore = kv)
```

#### <span id="13">Process</span>

We set epochs=100, batch-size=16

```python
for epoch in epochs:
	train_loss = 0.0
	for i, (data, target) in enumerate(train_data):
    	with autograd.record(True):
        	outputs = model(data)
        	losses = criterion(outputs, target)
        	mx.nd.waitall()
        	autograd.backward(losses)
    	optimizer.step(batch_size)
    	for loss in losses:
        	train_loss += loss.asnumpy()[0] / len(losses)
    	print('Epoch %d, batch %d, training loss %.3f'%(epoch, i, train_loss/(i+1)))
```

### <span id="14">Visualization</span>

```python
import random
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from gluoncv.utils.viz import get_color_pallete, DeNormalize
#select an image from trainset randomly
random.seed(datetime.now())
idx = random.randint(0, len(trainset))
img, mask = trainset[idx]
# get color pallete for visualize mask
mask = get_color_pallete(mask.asnumpy(), dataset='pascal_voc')
mask.save('mask.png')
# denormalize the image
img = DeNormalize([.485, .456, .406], [.229, .224, .225])(img)
img = np.transpose((img.asnumpy()*255).astype(np.uint8), (1, 2, 0))
# subplot 1 for img
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(img)
# subplot 2 for the mask
mmask = mpimg.imread('mask.png')
fig.add_subplot(1,2,2)
plt.imshow(mmask)
# display
plt.show()
```

![ss7](/img/ss/ss7.png)

### <span id="15">Experiment Condition</span>

**System**：Linux Ubuntu16.03

**GPU**：Nvidia GeForce 1080Ti×2

**Language**：Python3.6

**Framework**：MXnet1.3.1

## Summary

Through the practice of semantic segmentation, I have deepened my understanding of the specific application of computer vision，and I am full of expectations of instance segmentation and image segmentation that adopts traditional clustering methods. By using FCN, I realized the cleverness of different network structures designed for different tasks. Through the use of Dataparallel and SyncBN, I have mastered some skills to optimize and accelerate the training process. I hope I can contact more things in this field in the future and come up with some new ideas by myself.
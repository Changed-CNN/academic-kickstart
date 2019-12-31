+++

title= "Object detection and key point location based on ResNet"
date= 2019-08-28
tags=["DeepLearning", "CV", "ResNet", "CNN", "DeepFashion", "Location"]

+++

## Mission introduction

In object detection and marking, deep learning method is popular because of its excellent effect, This project was completed during my internship in `the key laboratory of speech and image processing` in the summer vacation of 2018. Before entering the lab, I had only been exposed to some traditional machine learning methods, so I did it while I was learning, and the Stanford course `CS231n` is highly recommended here.

What we need to solve is the precise positioning of the license plate in the picture containing the license plate after coarse positioning. After entering a picture containing the license plate, the four vertices of the license plate can be precisely located. Later, we also did an extended task, which was to complete `DeepFashion`'s clothing classification and key point positioning based on the code of license plate positioning.

## Outline

- **[Image Data Processing](#j0)**
- **[Network Model](#j1)**
  - **[CNN](#j2)**
  - **[ResNet](#j3)**
- **[Configuration Details](#j4)**
  - **[Optimization Settings](#j5)**
  - **[Loss Function](#j6)**
  - **[Pre-training](#j7)**
- **[Enhancement](#j8)**
- **[Training Process](#j9)**
- **[Evaluation Result](#j10)**
  - **[Numerical Result](#j11)**
  - **[Graphic Result](#j12)**
- **[Extension Work](#j13)**
- **[Experiment Condition](#j14)**

## Experiment Contents

### <span id="j0">Image Data Processing</span>

Our experimental data are not ready-made and processed, but are the property of the laboratory. The first thing we need to do is to manually cut out the approximate position of the license plate, and then mark the four vertices of the license plate in the picture to get the coordinates.

We obtained the picture of the front of the car containing the license plate by manually cutting the picture of the car with the marking tool, Since the aspect ratio of the license plate is roughly `2:1`, if we treat it as a normal picture and turn it into a `1:1` aspect ratio, the picture of the license plate will be distorted, so the final result of our processing is the picture with the aspect ratio of`2:1`, which will bring new problems to be solved when we build the network，for example, we need to control the size change of each layer inside the network. After preprocessing, we resize the image to 64*128. Then label the four vertices of the license plate on the processed image through the labeling tool, and obtain the final data set (image and label). The content of label is the horizontal and vertical coordinates of the four vertices corresponding to the image.

 In addition, we divided the data set of `1200` samples into training set with `1000` samples and test set with `200` samples.

### <span id="j1">Network Model</span>

#### <span id="j2">CNN</span>

**What Is CNN?**

> The name “convolutional neural network” indicates that the network employs a mathematical operation called [convolution](https://en.wikipedia.org/wiki/Convolution). Convolution is a specialized kind of linear operation. Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers. A convolutional neural network consists of an input and an output layer, as well as multiple [hidden layers](https://en.wikipedia.org/wiki/Multilayer_perceptron#Layers). The hidden layers of a CNN typically consist of a series of convolutional layers that *convolve* with a multiplication or other [dot product](https://en.wikipedia.org/wiki/Dot_product). The activation function is commonly a [RELU layer](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)), and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final [convolution](https://en.wikipedia.org/wiki/Convolution). The final convolution, in turn, often involves [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) in order to more accurately weight the end product.
>
> ——_From  Wikipedia_

**Structure Of Our CNN**

This model has eight layers. The first six layers are the same, including a `two-dimensional convolution layer`, `RELU` activation function, and the `two-dimensional maximum pooling layer `(refer to the internal structure order of `conv1`). The seventh layer is the full connection layer, including `Linear layer`, `RELU` activation function and `Dropout` to reduce overfitting. The eighth layer is also the full connection layer, which outputs the horizontal and vertical coordinates of the four vertices.

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 128, 64)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=32,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),  # output shape (32, 128, 64)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (32, 64, 32)
        )
        self.conv2 = nn.Sequential(  # input shape (32, 64, 32)
            nn.Conv2d(32, 64, 3, 1, 1),  # output shape (64, 64, 32)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 32, 16)
        )
        self.conv3 = nn.Sequential(  # input shape (64, 32, 16)
            nn.Conv2d(64, 128, 3, 1, 1),  # output shape (128, 32, 16)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (128, 16, 8)
        )
        self.conv4 = nn.Sequential(  # input shape (128, 16, 8)
            nn.Conv2d(128, 256, 3, 1, 1),  # output shape (256, 16, 8)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (256, 8, 4)
        )
        self.conv5 = nn.Sequential(  # input shape (256, 8, 4)
            nn.Conv2d(256, 512, 3, 1, 1),  # output shape (512, 8, 4)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (512, 4, 2)
        )
        self.conv6 = nn.Sequential(  # input shape (512, 4, 2)
            nn.Conv2d(512, 1024, 3, 1, 1),  # output shape (1024, 4, 2)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (1024, 2, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024 * 1 * 2, 1024),  # fully connected layer, output 1024
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(1024, 2 * 4)  # fully connected layer, output 4 * 2 points

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)  # flatten the output of convolution6 to (batch_size, 1024 * 2 * 1)
        fc = self.fc(x)
        output = self.out(fc)
        return output
```

#### <span id="j3">ResNet</span>

**What Is ResNet？**

> Residuals networks are characterized by ease of optimization and the ability to increase accuracy by increasing considerable depth. The internal residual block uses jump connections to alleviate the problem of gradient disappearance caused by increasing depth in the deep neural network. The traditional convolutional network or fully connected network has more or less problems such as information loss during information transmission, and it also leads to gradient disappearance or gradient explosion, which makes it impossible to train deep networks. ResNet solves this problem to a certain extent by directly bypassing the input information to the output to protect the integrity of the information. The whole network only needs to learn the part of the difference between input and output to simplify the learning objectives and difficulties.
>  ——_From  CSDN_

The neural network used before was the most basic `CNN`, and the effect was quite good. However, in order to further improve the effect, we used `ResNet` for training, and the effect was rapidly improved.

**Structure Of Our ResNet**

On the basis of ResNet-18, our model was improved according to the experimental conditions, adding another layer of convolution layer after the original four layers of convolution layer, and changing the average pooling layer to the maximum pooling layer. A BatchNorm in the outer network was deleted and two full connection layers were added at the end of the network. The output is the horizontal and vertical coordinates of four vertices, and Batchsize is 16, so the output is 16*8.

We know that ResNet is made up of basic blocks, whose code is shown below:

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
```

The code for the overall structure is shown below:

```python
class ResNet_2(nn.Module):
    def __init__(self, block, layers, num_classes=8):
        self.inplanes = 64
        super(ResNet_2, self).__init__()
        #input = 3*64*128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)#output = 32*32*64
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)#output = 32*32*64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=2)
        self.avgpool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024,2*4)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
```

### <span id="j4">Configuration Details</span>

#### <span id="j5">Optimization Settings</span>

We used Adam gradient descent method and dynamic learning rate to optimize.

From the `1st` epoch to the `49th` epoch, the learning rate was set as `0.01×0.9^epoch` and updated every ten epochs. From the `49th` epoch to the `65th` epoch, the learning rate was fixed at `0.01×0.9^40`.

```python
for epoch in range(1, args.epochs + 1):
        begin = time()
        if epoch % 10 == 0:
            optimizer = optim.Adam(model.parameters(), lr=0.01 * pow(0.9, epoch))
            if epoch > 49:
                optimizer = optim.Adam(model.parameters(), lr=0.01 * pow(0.9, 40))
```

#### <span id="j6">Loss Function</span>

In training we adopt `mean square error` (`MSE Loss`/`L2 Loss`), the mean square error of all feature points was calculated and the mean value was taken as the training loss.

```python
train_loss += F.mse_loss(output, target).item()  # sum up batch loss
return train_loss / len(train_loader)
```

In test we adopt `mean absolute error`(`MAE Loss`/`L1 Loss`), the mean absolute error of all feature points was calculated and the mean value was taken as the training loss.

```python
test_loss += F.l1_loss(output, target).item() # sum up batch loss
return test_loss / len(test_loader)
```

#### <span id="j7">Pre-training</span>

If pre-training is required, load the molded network parameters from the online model zoo

```python
if pretrained:   model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
```

### <span id="j8">Enhancement</span>

A random number is set to `rotate` the image at random angles between (-20, 20)

```python
def rotate(image, angle, center= None, scale=1.0):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
def RGBT(image):
    ran = random.randint(-20, 20)
    image = rotate(image, ran)
    image = Image.fromarray(cv2.cvtColor(image, 1))
    return image, ran
```

`Gaussian blur` is used to improve generalization ability

```python
def gaussBlur(image):
    kernel_size = (9, 9)
    sigma = 40
    img = cv2.GaussianBlur(image, kernel_size, sigma)
    return img
```

### <span id="j9">Training Process</span>

epoch=65    batch_size=16

![cp](/img/cp/cp.jpg)

![cp-1](/img/cp/cp-1.jpg)

### <span id="j10">Evaluation Result</span>

#### <span id="j11">Numerical Result</span>

Evaluation basis: The image's `bounding box` width 128 is adopted as the standard of normalized evaluation, and average the distance between the four points of output and the four points of label, then divide the average value by the width as the evaluation basis.

```python
d1 = sqrt(square(i[0][0] - j[0][0]) + square(i[0][1] - j[0][1]))
d2 = sqrt(square(i[0][2] - j[0][2]) + square(i[0][3] - j[0][3]))
d3 = sqrt(square(i[0][4] - j[0][4]) + square(i[0][5] - j[0][5]))
d4 = sqrt(square(i[0][6] - j[0][6]) + square(i[0][7] - j[0][7]))
d.append(float((d1 + d2 + d3 + d4)/4))
print('Average evaluate:',sum(d) / (128*batch_idx))
```

![cp-2](/img/cp/cp-2.jpg)

#### <span id="j12">Graphic Result</span>

The pink point is the ground truth, and the red line is connected by the output points. The object can also be detected under partial occlusion. If some points are not in the picture, it can be judged according to the coordinates of the output points. If the points are outside the coordinate area, it will not be displayed.

```python
i = output.cpu().detach().numpy()
j = batch['landmarks'].cpu().detach().numpy()
plt.imshow(tensor_to_PIL(batch['image'][0]))
plt.xlim(0, 128)
plt.ylim(64, 0)
plt.plot([i[0][0], i[0][2]], [i[0][1], i[0][3]], color='r')
plt.plot([i[0][2], i[0][4]], [i[0][3], i[0][5]], color='r')
plt.plot([i[0][4], i[0][6]], [i[0][5], i[0][7]], color='r')
plt.plot([i[0][6], i[0][0]], [i[0][7], i[0][1]], color='r')
plt.scatter(j[0][0], j[0][1], color = 'pink')
plt.scatter(j[0][2], j[0][3], color = 'pink')
plt.scatter(j[0][4], j[0][5], color = 'pink')
plt.scatter(j[0][6], j[0][7], color = 'pink')
```

![cp-3](/img/cp/cp-3.jpg)

### <span id="j13">Extension Work</span>

**DeepFashion's clothing classification and key point positioning**

FashionNet's forward computing process is divided into three stages: In the first stage, we input a picture of clothes into a branch of the network to predict whether the key points of clothes are visible and their specific positions. In the second stage, local features of clothes are obtained by the landmark pooling layer according to the predicted locations of the key points in the previous step. In the third stage, the global features of "fc6_global" layer and local features of "fc6_local" are spliced together to form "fc7_fusion" as the final image features.

![cp-4](/img/cp/cp-4.jpg)

Our task is simpler than the task in this paper, we modified the code of key point positioning of license plate to complete the basic identification and key point positioning of clothing. Because the number of key points of different types of clothing is different, it is necessary to identify the clothing first, so as to determine the output data specification according to the clothing type.

You can get more details about FashionNet from [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html).

### <span id="j14">Experiment Condition</span>

System：Linux Ubuntu16.03

GPU：Nvidia GeForce 1080Ti

Language：Python3.6

Framework：PyTorch 0.4

## Summary

> Through the joint efforts of my teammates in a summer vacation, I entered the door of deep learning and computer vision. From the beginning, I could not do anything until I was able to skillfully optimize it, At last, we successfully completed the tasks assigned by the professor and got the professor's approval. At the same time, it has aroused my strong interest in the field of deep learning and computer vision, and I hope to do more research in related fields in the future.
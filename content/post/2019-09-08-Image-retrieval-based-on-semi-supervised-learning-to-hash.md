+++
title="Image retrieval based on semi-supervised learning to hash"
date=2019-09-08
tags=["Hash", "Pytorch", "Semi-supervised", "ICT", "DHD", "MAP"]

+++

## Introduce

Hashing is a widely used technique to the task of approximate nearest neighbor search, which aims at transforming the high dimensional data to a set of short hash codes. Learning to hash aims to learn a hash function from the given data, which can preserve the similarity of original data under the Hamming Space.  Supervised deep neural networks are applied to achieve it, but the effect is not very good, so we decided to use semi-supervised method to improve the effect. And we combined it with  `Interpolation Consistency Training` (ICT) and `Discriminative Hashing basd on angularly Discriminative embedding`(DHD), these two points are from two papers. This project was completed during internship in the Center of Optical Image Analysis&Learning and produced a journal paper.

## Ideas

### **DHD**

Supervised deep hashing aims to learn a nonlinear hash function F : x → b ∈{−1,1} that can map each xi into a k-bit binary code bi. The ideal hash codes are expected to have smaller intra-class distance and larger inter-class distance under Hamming Space.  For each sample x, we obtain its hash code by performing the sign function for continuous embedding f returned by the trained model.

Pipeline:

![hash1](/img/hash/hash1.png)

The main network can be any convolutional neural network without the classiﬁer layer, such as AlexNet. We add an additional fully connected layer to map the features D as embeddings F. In addition, the objective is a weighted sum of the A-Softmax Loss and a Regularization term over embeddings F. To obtain the ﬁnal hash codes, we perform the sign function for F returned by the trained model.

### **ICT**

ICT encourages the prediction at an interpolation of unlabeled points to be consistent with the interpolation of the predictions at those points. In classiﬁcation problems, ICT moves the decision boundary to low-density regions of the data distribution. The goal of Semi-Supervised Learning (SSL) is to leverage large amounts of unlabeled data to improve the performance of supervised learning over small datasets. ICT regularizes semi-supervised learning by encouraging consistent predictions `f(αu1 + (1−α)u2) = αf(u1)+(1−α)f(u2)` at interpolations `αu1 +(1−α)u2` of unlabeled points `u1` and `u2`. 

Pipeline:

![hash2](/img/hash/hash2.png)

ICT learns a student network fθ in a semi-supervised manner. To this end, ICT uses a mean-teacher fθ‘, where the teacher parameters θ’ are an exponential moving average of the student parameters θ. During training, the student parameters θ are updated to encourage consistent predictions `fθ(Mixλ(uj,uk)) ≈Mixλ(fθ’(uj),fθ’(uk))`, and correct predictions for labeled examples xi.

### **Combination**

The purpose of our experiment is to realize image retrieval based on semi-supervised learning to generate hash codes. We use the network structure in DHD and make some adjustments to make it possible to generate hash codes, combined with the semi-supervised learning process in ICT. This will have a better effect than supervised learning to generate hash codes, the generated hash codes will have smaller intra-class distance and larger inter-class distance under Hamming Space.

## Outline

- [Data Processing](#1)
- [Networks](#2)
- [Details](#3)
- [Evaluation](#4)
- [Operations](#5)
- [Environment](#6)

## Contents

### <span id="1">Data Processing</span>

There is no `ZCA` in the preprocessing of DHD. `ZCA` in ICT does improve the accuracy of `topk` to some extent, but has a great impact on the evaluation standard `MAP` introduced in DHD. Therefore, we finally remove `ZCA` in the preprocessing.

```python
def apply_zca(data, zca_mean, zca_components):
        temp = data.numpy()
        shape = temp.shape
        temp = temp.reshape(-1, shape[1]*shape[2]*shape[3])
        temp = np.dot(temp - zca_mean, zca_components.T)
        temp = temp.reshape(-1, shape[1], shape [2], shape[3])
        data = torch.from_numpy(temp).float()
        return data
```

The first step is to preprocess the content of ImageNet data.

```python
train_transform_imagenet = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform_imagenet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

Since semi-supervised learning is required, we need to divide the complete data set into two data sets. We select one of every ten pictures in the complete dataset as labeled data, and the remaining nine are unlabeled data. Each image in the txt file of the dataset occupies one line, and each line consists of the path of the image and its hash code label.

```python
i=1
file_to_write_train=open('E:/dataset/train.txt', 'w')
file_to_write_unlabelled=open('E:/dataset/unlabelled.txt', 'w')
with open('E:/dataset/database.txt','r') as file_to_read:
    while True:
        lines = file_to_read.readline()
        if not lines:
            break
        if i%10==1:
            file_to_write_train.writelines(lines)
        else:
            file_to_write_unlabelled.writelines(lines)
        i = i + 1
file_to_write_train.close()
file_to_write_unlabelled.close()
```

The next step is the partition operation of labeled training set, test set, unlabeled training set and the generation of dataloaders.

```python
train_dataset_imagenet = IMAGENET(root='data/imagenet/train.txt',
                         train=True,
                         transform=train_transform_imagenet)
test_dataset_imagenet = IMAGENET(root='data/imagenet/test.txt',
                        train=False,
                        transform=test_transform_imagenet)
database_dataset_imagenet = IMAGENET(root='data/imagenet/dataset.txt',
                            train=False,
                            transform=test_transform_imagenet,
                            database_bool=True)
unlabelled_dataset_imagenet = IMAGENET(root='data/imagenet/unlabelled.txt',
                            train=False,
                            transform=train_transform_imagenet,
                            unlabelled=True)
train_loader_imagenet = torch.utils.data.DataLoader(dataset=train_dataset_imagenet,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=8)
test_loader_imagenet = torch.utils.data.DataLoader(dataset=test_dataset_imagenet,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=8)
database_loader_imagenet = torch.utils.data.DataLoader(dataset=database_dataset_imagenet,
                                              batch_size=32,
                                              shuffle=True,
                                              num_workers=8)
unlabelled_loader_imagenet = torch.utils.data.DataLoader(dataset=unlabelled_dataset_imagenet,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=8)
```

### <span id="2">Networks</span>

We replaced the `AngleLoss` in DHD with the ordinary `SoftmaxLoss`, and added a nonlinear activation layer `tanh` after the feature layer to compress the data into the interval of -1 to 1, and then obtained the hash code by `sign` function. In the previous network structure, we used `HashCNN` and `CNN13`, where HashCNN is an extension of pre-trained `Alexnet`, while CNN13 is not pre-trained `CNN`.

**HashCNN**

```python
class hash_CNN(nn.Module):
    def __init__(self, num_classes=10,dropout=0.0,bit=64):
        super(hash_CNN, self).__init__()
        self.alex = torchvision.models.alexnet(pretrained=True)
        self.drop1 = nn.Dropout(dropout)
        self.alex.classifier = nn.Sequential(*list(self.alex.classifier.children())[:6])
        self.layer = nn.Sequential(
            nn.Linear(4096, bit),
            nn.Tanh()
         )
        self.margin = ArcMarginProduct(bit, num_classes)
    def forward(self, x):
        x = self.alex.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.alex.classifier(x)
        x = self.layer(x)
        code = hash_layer(x)
        output = self.margin(x)
        return output, code, x
```

**CNN13**

```python
class CNN13(nn.Module):      
    def __init__(self, num_classes=10, dropout=0.5,bit=64):
        super(CNN13, self).__init__()
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(dropout)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(dropout)
        
        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 =  weight_norm(nn.Linear(128, num_classes))
        self.layer = nn.Sequential(
            nn.Linear(128, bit),
            nn.Tanh()
        )
        self.fc = weight_norm(nn.Linear(bit, num_classes))
```

_Tips: The declaration of the network model must be a global variable. If it is a local variable, it is likely that the network tested and trained is not the same._

### <span id="3">Details</span>

**Loss**

We replaced `AngleLoss` in DHD with `CrossEntropyLoss` to calculate **`class_loss`**. After replacement, some problems occurred in details, mainly caused by the different return values of two losses. Therefore, we deleted all relevant codes about getting the 0th dimension of loss to solve them. **`mixup consistency loss`** is an interpolation consistency regularization term and its weight is constantly increasing after each iteration, but it is only used when semi-supervised learning. **`loss2`** is also used as a regularizer to enhance anti-disturbance capability and prevent over-fitting.

```python
#class loss
class_criterion = nn.CrossEntropyLoss().cuda()
class_loss = class_criterion(class_logit, target_var) / minibatch_size
#mixup consistency loss
consistency_criterion = losses.softmax_mse_loss
mixup_consistency_loss = consistency_criterion(output_mixed_u, mixedup_target) / minibatch_size
mixup_consistency_weight = get_current_consistency_weight(args.mixup_consistency, epoch, i, len(unlabelledloader))
mixup_consistency_loss = mixup_consistency_weight*mixup_consistency_loss
#loss2
loss2 = torch.mean(torch.abs(torch.pow(torch.abs(feature)- Variable(torch.mean(torch.abs(feature), dim=1, keepdim=True).repeat(1, 64).cuda()), 3)))
#Loss
Loss = class_loss + mixup_consistency_loss + 0.3 * loss2
```

**Learning Rate**

Assume that the initial learning rate is 0.1, after 30 epochs it will change to 0.01, and it will change to 0.001 after 80 epochs. It's a widely used learning rate adjustment strategy.

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
```

**Optimizer**

```python
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
```

### <span id="4">Evaluation</span>

**Accuracy**

Directly calculate the percentage of the number of correctly predicted hash code labels

```python
		model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader_imagenet:
            images = Variable(images.cuda())
            out1, _, _ = model(images)
            _, predicted = torch.max(out1.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Test Accuracy of the model: %.2f %%' % (100.0 * correct / total))
```

**MAP**

Get the hash code label of images and the hash code output through the network

```python
def binary_output(dataloader):
    model.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
        targets = Variable(targets)
        out, code, _ = model(inputs)
        full_batch_output = torch.cat((full_batch_output, code.data), 0)
        full_batch_label = torch.cat((full_batch_label, targets.data), 0)
    return torch.sign(full_batch_output), full_batch_label
```

Mean Average Precision(MAP) is a common precision measurement standard in image retrieval task

```python
def test_map():
    model.cuda()
    model.eval()
    retrievalB, retrievalL = binary_output(database_loader_imagenet)
    queryB, queryL = binary_output(test_loader_imagenet)
    map = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
    print("map: ", map)
def calculate_map(qB, rB, queryL, retrievalL):
    num_query = queryL.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (retrievalL.cpu() == queryL[iter].cpu()).numpy().astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        nn = qB[iter, :].cpu().numpy()
        mm = rB.cpu().numpy()
        hamm = 0.5 * (mm.shape[1] - np.dot(nn, mm.T))
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)  
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map
```

**Effect**

After replacing the network, we did not pay attention to the differences in the training process of labeled data, which resulted in a high precision while the MAP always failed to meet expectation. We mistakenly thought that it was CNN13 that caused the bad effect, because the DHD itself had excellent effect, so we replaced the network with the pre-trained HashNet in DHD. The final problem is that ICT interpolates labeled data, which is the main reason why MAP can't be promoted. After removing it, a higher mAP is produced.

>Two experiments were conducted on Cifar10 based on `HashNet` with pre-training:
>
>1. Supervised learning without consistency_loss
>2. Semi-Supervised learning with consistency_loss

But the MAP wasn't good enough, so Loss2 was added and two more experiments were conducted, then the ideal value of Cifar10's MAP is achieved. In the case of supervised learning, `MAP ≈0.810`; in the case of semi-supervised learning, `MAP ≈0.836`.

But the effect of semi-supervisied learning is not much higher than that of supervised learning, there is a misconception that pre-training network must produce better results. In fact, it is wrong, because the data used in the pre-training process may not be the data in Cifar10, and some parameters are not adjusted to better fit Cifar10. However, it does not affect the final convergence, and the convergence rate is much higher than that of CNN13 without pre-training. Therefore, the large amount of data used in the pre-training process leads to a small gap between supervised learning and semi-supervised learning. 

>Two experiments were conducted on Cifar10 based on `CNN13` without pre-training:
>
>1. Supervised learning with loss2 but without consistency_loss
>2. Semi-Supervised learning with loss2 and consistency_loss

There is a large gap between the two MAP. In the case of supervised learning, `MAP ≈0.71`; in the case of semi-supervised learning, `MAP ≈0.866`. Since pre-training makes up for the disadvantage of small amount of supervised learning data, and loses this advantage in CNN13, the advantage of semi-supervised learning itself appears, and the effect is even better than that of pre-training network HashNet.

Similar experiments with ImageNet are underway......

### <span id="5">Operations</span>

**Tools&Commands**

We use `Pycharm` to edit the code in the Windows environment, then transfer it to the server using `Xftp6`, and remotely control the server with `Xshell6`. In the future, I plan to use `Pycharm+Docker+Tensorflow` for deep learning, so that I can do everything directly in the editor.

First we need to activate the environment needed to run the experiment

```shell
source activate /home/hsz/anaconda2
```

Then we need to run the code with instructions that contain hyperparameters

```shell
python main.py  --dataset cifar10  --num_labeled 500 --num_valid_samples 500 --root_dir experiments/ --data_dir data/cifar10/ --batch_size 32  --arch hash_cnn --dropout 0.0 --mixup_consistency 100.0 --pseudo_label mean_teacher  --consistency_rampup_starts 0 --consistency_rampup_ends 100 --epochs 401  --lr_rampdown_epochs 450 --print_freq 200 --momentum 0.9 --lr 0.001 --ema_decay 0.999  --mixup_sup_alpha 0 --mixup_usup_alpha 1.0 
```

**Parameter Setting**

In fact there are many parameters that do not work, the main parameters we can modify are: `dataset`(Used to select Cifar10 or ImageNet or others)

`batch_size`(Used to set the size of per batch)

`arch`(Used to select the network)

`pseudo_label`(Used to control semi-supervised learning or supervised learning)

`epochs`(Used to set the number of training rounds)

`lr`(Used to set initial learning rate)

### <span id="6">Environment</span>

**System**：Windows 10/Linux

**GPU**：Nvidia GeForce TitanV

**Language**：Python 3.6

**Framework**：PyTorch 1.0.1

## Summary

In this project, my teammate and I were mainly responsible for the experimental part and experienced the whole process of creating a paper. First of all, we read about ten related papers, and then clarified our ideas and innovation points. When we just started to do experiments, we encountered difficulties in code integration, and later encountered various problems and fell into some misunderstandings. Therefore, the whole process was to constantly find and solve problems. After the whole process, I learned a lot of new knowledge and improved my scientific research ability. At the same time, I realized that I still had many shortcomings. However, after the constant test of these projects, I feel that I can face all kinds of problems. While marveling at the clever ideas, I also hope that I can come up with good ideas one day.
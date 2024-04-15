# -*- coding: utf-8 -*-
"""
# 1. Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# %matplotlib inline

"""**ResNet Model**"""

MODELNAME = "cifar.model"
EPOCH = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
# 2.Load Data

**Train - Test**
"""

train_dataset = CIFAR10(root='./', train=True,
                      transform=tv.transforms.ToTensor(),
                      download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

test_dataset = CIFAR10(root='./', train=False,
                     transform=tv.transforms.ToTensor(),
                     download = True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy() # convert images to numpy for display
fig = plt.figure(figsize=(25, 4))

import torch.nn as nn

class Block(nn.Module):
  def __init__(self, input_channels, output_channels, stride=1):
    super(Block, self).__init__()
    self.block = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=1, bias = False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, stride=stride, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels*4, kernel_size = 1, bias=False),
        nn.BatchNorm2d(output_channels*4),
    )

    self.shortcut = nn.Sequential()

    if stride != 1 or input_channels != output_channels * 4:
      self.shortcut = nn.Sequential(
          nn.Conv2d(input_channels, output_channels * 4, stride = stride, kernel_size = 1, bias = False),
          nn.BatchNorm2d(output_channels *4)
      )

  def forward(self, x):
    # print(x.shape)
    a = self.block(x)
    # print(a.shape)

    b = self.shortcut(x)
    # print(b.shape)

    return nn.ReLU(inplace=True)(a + b)

class ResNet(nn.Module):
  def __init__(self, block, num_block):
    super(ResNet, self).__init__()
    self.in_channels = 64

    self.conv1 = nn.Sequential(
        nn.Conv2d(3,64, kernel_size = 3, padding =1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )
    self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
    self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
    self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
    self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

    self.pool = nn.AdaptiveAvgPool2d((1,1))

    self.fc = nn.Linear(512*4, 10)
  def forward(self, x):
    # print(x.shape)
    output = self.conv1(x)
    # print(output.shape)
    output = self.conv2_x(output)
    output = self.conv3_x(output)
    output = self.conv4_x(output)
    output = self.conv5_x(output)
    output = self.pool(output)
    output = output.view(output.size(0), -1)
    output = self.fc(output)
    return output

  def _make_layer(self, block, out_channels, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_channels, out_channels, stride))
      self.in_channels = out_channels * 4

    return nn.Sequential(*layers)

"""# Train"""

def train():
  model = ResNet(Block, [3,4,6,3]).to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters())
  for epoch in range(10):
      train_loss = 0.0

      for images, label in train_loader:
        images, label = images.view(-1, 3, 32, 32).to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        output = model(images).to(DEVICE)

        batchloss = F.cross_entropy(output,label)
        batchloss.backward()
        optimizer.step()
        train_loss += batchloss.item()
      print("epoch", epoch, ": loss", train_loss)


  torch.save(model.state_dict(), MODELNAME)

train()

"""# Test"""

def test():
  correct = 0
  total = len(test_loader.dataset)
  model = ResNet(Block, [3,4,6,3]).to(DEVICE)
  model.load_state_dict(torch.load(MODELNAME))
  model.eval()
  for images, labels in test_loader:
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    y = model(images).to(DEVICE)
    pred_labels = y.max(dim=1)[1]
    correct = correct + (pred_labels == labels).sum()

  print("correct: ", correct.item())
  print("total: ", total)
  print("accuracy: ", correct.item()/total)

test()

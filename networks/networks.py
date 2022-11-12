import timm
import torch.nn.functional as F
import torch.nn as nn
class CNN_Model(nn.Module):
    def __init__(self, emb_size=512):
      super().__init__()
      self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
      self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
      self.batchnorm1 = nn.BatchNorm2d(64)
      self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
      self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3 ,padding=1)
      self.batchnorm2 = nn.BatchNorm2d(128)
      self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
      self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
      self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
      self.batchnorm3 = nn.BatchNorm2d(256)
      self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
      self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
      self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
      self.batchnorm4 = nn.BatchNorm2d(512)
      self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
      self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
      self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
      self.batchnorm5 = nn.BatchNorm2d(512)

      self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
      
      self.fc1 = nn.Linear(25088, 4096)


    def forward(self, images):
      x = F.relu(self.conv1_1(images))
      x = F.relu(self.conv1_2(x))
      x = self.batchnorm1(x)
      x = self.maxpool(x)
      x = F.relu(self.conv2_1(x))
      x = F.relu(self.conv2_2(x))
      x = self.batchnorm2(x)
      x = self.maxpool(x)
      x = F.relu(self.conv3_1(x))
      x = F.relu(self.conv3_2(x))
      x = F.relu(self.conv3_3(x))
      x = self.batchnorm3(x)
      x = self.maxpool(x)
      x = F.relu(self.conv4_1(x))
      x = F.relu(self.conv4_2(x))
      x = F.relu(self.conv4_3(x))
      x = self.batchnorm4(x)
      x = self.maxpool(x)
      x = F.relu(self.conv5_1(x))
      x = F.relu(self.conv5_2(x))
      x = F.relu(self.conv5_3(x))
      x = self.batchnorm5(x)
      x = self.maxpool(x)
      x = x.view(x.size(0), -1)#flatten
      x = F.relu(self.fc1(x))
      return x

            # x = self.conv1(images)
      # x = F.relu(x)
      # x = self.conv2(x)
      # x = F.relu(x)
      # x = self.batchnorm1(x)
      # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)


      return x
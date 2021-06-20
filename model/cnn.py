import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.mp1 = nn.MaxPool2d(4)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.mp2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d(2)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        
        self.amp = nn.AdaptiveMaxPool2d((4,4))
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        #self.fc2 = nn.Linear(1024, 512)
        self.bn1d = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1,bias=False)
        
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x,inplace = True)
        x = self.conv2(x)
        x = F.relu(x,inplace = True)
        x = self.conv3(x)
        x = F.relu(x,inplace = True)
        x = self.mp1(x)
        
        x = self.conv4(x)
        x = F.relu(x,inplace = True)
        x = self.bn1(x)
        x = self.mp2(x)
        x = self.conv5(x)
        x = F.relu(x,inplace = True)
        x = self.bn2(x)
        x = self.mp3(x)
        x = self.conv6(x)
        x = F.relu(x,inplace = True)
        x = self.amp(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=0.5)
        #x = self.fc2(x)
        emb = self.bn1d(x)
        x = self.fc2(x)
        
        return x,emb


        



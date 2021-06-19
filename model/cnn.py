import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(4)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(96)
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d(2)
        
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1)
        self.bn9 = nn.BatchNorm2d(128)
        self.amp = nn.AdaptiveMaxPool2d((8,8))
        
        self.fc1 = nn.Linear(128 * 8 * 8, 4048)
        self.fc2 = nn.Linear(4048, 512)
        self.fc3 = nn.Linear(512, 1,bias=False)
        
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.mp1(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.bn6(x)
        x = self.mp2(x)
        
        x = self.conv7(x)
        x = F.relu(x)
        x = self.bn7(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.bn8(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.bn9(x)
        x = self.amp(x)
        
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        emb = F.relu(self.fc2(x))
        x = self.fc3(emb)
        
        return x,emb




import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)  # (b x 96 x 55 x 55) 
        self.relu = nn.ReLU()
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)  # section 3.3
        self.maxpool1 =nn.MaxPool2d(kernel_size=3, stride=2)  # (b x 96 x 27 x 27)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)  # (b x 256 x 27 x 27)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # (b x 256 x 13 x 13)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)  # (b x 384 x 13 x 13)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)  # (b x 384 x 13 x 13)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)  # (b x 256 x 13 x 13)
      
        self.adpmaxpool = nn.AdaptiveMaxPool2d((6,6))
        self.drop1 = nn.Dropout(p=0.5, inplace=True)
        self.linear1 = nn.Linear(in_features=(256 * 6 * 6), out_features=4096)
        
        self.drop2 = nn.Dropout(p=0.5, inplace=True)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=num_classes)
  
        self.sigmoid = nn.Sigmoid()
    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.lrn1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.adpmaxpool(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.drop2(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return self.sigmoid(x)





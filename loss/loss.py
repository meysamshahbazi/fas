import torch.nn as nn
import torch
import torch.nn.functional as F


class ArcB(nn.Module):
    def __init__(self,net, m=0.5,s=1.0):
        super(ArcB, self).__init__()
        print("*************")
        self.m = m 
        self.s = s
        self.net = net


    def forward(self,outputs,classes,emb):
        w = list(self.net.parameters())[-1]
        w = F.normalize(w)
        emb = F.normalize(emb)
        theta = torch.acos(emb.matmul(w.T))
        l = -classes*torch.log(1/(1+torch.exp(-self.s*torch.cos(theta+self.m)))) -\
        (1-classes)*torch.log(1-1/(1+torch.exp(-self.s*torch.cos(theta-self.m))))
        return l.mean()



class BCEWithLogits(nn.Module):
    def __init__(self):
        super(BCEWithLogits,self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,outputs,classes,emb):
        return self.criterion(outputs, classes)


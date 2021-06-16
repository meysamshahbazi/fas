import torch.nn as nn
import torch



class ArcB(nn.Module):
    def __init__(self,net, m=0.5):
        super(ArcB, self).__init__()
        print("*************")
        self.m = m 
        self.net = net


    def forward(self,outputs,classes,emb):
        w = list(self.net.parameters())[-2]
        b = list(self.net.parameters())[-1]
        s = torch.norm(w.T)*torch.norm(emb,dim=1)
        s = s.unsqueeze(dim=1)
        theta = torch.acos((emb.matmul(w.T))/s)# take care about sign of m !!
        l = -classes*torch.log(1/(1+torch.exp(-s*torch.cos(theta-self.m)-b))) -\
        (1-classes)*torch.log(1-1/(1+torch.exp(-s*torch.cos(theta+self.m)-b)))
        return l.mean()

class BCEWithLogits(nn.Module):
    def __init__(self):
        super(BCEWithLogits,self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,outputs,classes,emb):
        return self.criterion(outputs, classes)


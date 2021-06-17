import torch.nn as nn
import torch
import torch.nn.functional as F
import itertools

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
        (1-classes)*torch.log(1-1/(1+torch.exp(-self.s*torch.cos(theta+self.m))))
        #l = -classes*torch.log(1/(1+torch.exp(-self.s*torch.cos(theta+self.m)))) -\
        #(1-classes)*torch.log(1/(1+torch.exp(-self.s*torch.cos(theta-self.m))))
        self.net.linear3.weight = torch.nn.Parameter(w*self.s) 
        #list(self.net.parameters())[-1] = w*self.s
        return l.mean()



class BCEWithLogits(nn.Module):
    def __init__(self):
        super(BCEWithLogits,self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,outputs,classes,emb):
        return self.criterion(outputs, classes)


class IdBce(nn.Module):
    def __init__(self,alpha=0.5,M=0.5):
        super(IdBce,self).__init__()
        self.alpha = alpha
        self.M = M
        self.bce =  nn.BCEWithLogitsLoss()

    def forward(self,outputs,classes,emb,ids):
        l = 0
        emb = F.normalize(emb)
        for i,j in itertools.combinations(range(len(ids)), 2):
            l+= 0.5*int(ids[i]==ids[j]and classes[i]!=classes[j])*((emb[i]-emb[j])**2).sum()
            l+=int(ids[i]!=ids[j]and classes[i]==classes[j])*torch.max(torch.FloatTensor([0,self.M-((emb[i]-emb[j])**2).sum()]))
        print("l is:",l)
        return self.bce(outputs, classes) + self.alpha*l

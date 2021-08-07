import torch.nn as nn
import torch
import torch.nn.functional as F
import itertools
import math

class ArcB(nn.Module):
    def __init__(self,net, m=0.5,s=1.0):
        super(ArcB, self).__init__()
        print("*************")
        self.m = m 
        self.s = s
        self.net = net
        self.bce = nn.BCEWithLogitsLoss()


    def forward(self,outputs,classes,emb,ids):
        w = list(self.net.parameters())[-1]
        scale = torch.norm(w)*torch.norm(emb,dim=1)
        scale = scale.unsqueeze(dim=1)
        theta = torch.acos(emb.matmul(w.T)/scale)
        outs = classes*scale*torch.cos(theta+self.m) + (1-classes)*scale*torch.cos(theta-self.m)
        return self.bce(outs,classes)



class BCEWithLogits(nn.Module):
    def __init__(self):
        super(BCEWithLogits,self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,outputs,classes,emb,ids):
        return self.criterion(outputs, classes)

'''
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
            l+=int(ids[i]!=ids[j]and classes[i]==classes[j])*torch.max(torch.FloatTensor([0,self.M-(torch.abs(emb[i]-emb[j])).sum()]))
        #print("l is:",l)
        return self.bce(outputs, classes) + self.alpha*l

'''

class IdBce(nn.Module):
    def __init__(self,alpha=0.5,M=0.5):
        super(IdBce,self).__init__()
        self.alpha = alpha
        self.M = M
        self.bce =  nn.BCEWithLogitsLoss()

    def forward(self,outputs,classes,emb,ids):
            
        idx1 = [] #index for same id but diffrent class label
        idx2 = [] #index for same class label but diffrent id
        l = 0 
        emb = F.normalize(emb)

        idd = torch.combinations(ids.reshape(-1),2)
        cll = torch.combinations(classes.reshape(-1),2)
        idxx = torch.combinations(torch.arange(0,len(ids)),2)
        c1 = torch.logical_and(idd[:,0]==idd[:,1],cll[:,0]!=cll[:,1]).reshape(-1,1) # index for same id but diffrent class label
        c2 = torch.logical_and(idd[:,0]!=idd[:,1],cll[:,0]==cll[:,1]) # index for same class label but diffrent id
        emb_dif = emb[idxx[:,0]]-emb[idxx[:,1]]
        if c1.sum() > 0: 
            l += torch.norm((c1*emb_dif),dim=1).sum()/c1.sum()
        if c2.sum() > 0:
            l += ( c2*torch.max(torch.zeros(len(idxx),device=emb.device),self.M-torch.norm((emb_dif),2,dim=1)) ).sum()/(c2.sum())
        lbce= self.bce(outputs, classes)
        # print(lbce," | ",l)
        return lbce + self.alpha*l




class ArcbId(nn.Module):

    def __init__(self,alpha,beta,net,M = 0.5,m=0.5):
        super(ArcbId,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.M = M
        self.bce = nn.BCEWithLogitsLoss()
        self.net =  net
        self.m = m
        print("ArcB loss with Person ID based loss")

    def forward(self,outputs,classes,emb,ids):
        w = list(self.net.parameters())[-1]
        scale = torch.norm(w)*torch.norm(emb,dim=1)
        scale = scale.unsqueeze(dim=1)
        theta = torch.acos(emb.matmul(w.T)/scale)
        # theta1 = torch.where(theta<math.pi/2-self.m,theta,theta+self.m)
        # theta2 = torch.where(theta>math.pi/2+self.m,theta,theta-self.m)
        outs = classes*scale*torch.cos(theta+self.m) + (1-classes)*scale*torch.cos(theta-self.m)
        # outs = classes*scale*torch.cos(theta1) + (1-classes)*scale*torch.cos(theta2)
        l = 0

        emb = F.normalize(emb)

        idd = torch.combinations(ids.reshape(-1),2)
        cll = torch.combinations(classes.reshape(-1),2)
        idxx = torch.combinations(torch.arange(0,len(ids)),2)
        c1 = torch.logical_and(idd[:,0]!=idd[:,1],cll[:,0]==cll[:,1]).reshape(-1,1) # index for same class but diffrent id
        c2 = torch.logical_and(idd[:,0]==idd[:,1],cll[:,0]!=cll[:,1]) # index for difrrent class label but same id
        emb_dif = emb[idxx[:,0]]-emb[idxx[:,1]] 
        l1 = 0
        l2 = 0
        if c1.sum() > 0: 
            l1 = torch.norm((c1*emb_dif),dim=1).sum()/c1.sum()

        if c2.sum() > 0:
            # l2 = torch.norm((c2*emb_dif),dim=1).sum()/c2.sum()
            l2 = ( c2*torch.max(torch.zeros(len(idxx),device=emb.device),self.M-torch.norm((emb_dif),2,dim=1)) ).sum()/(c2.sum())
        # l = l1+l2
        lbce = self.bce(outs,classes)
        # print(str(l1)," | ",str(l2) , " | ",lbce)
        return lbce + self.alpha*l1 + self.beta*l2



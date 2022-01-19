import torch.nn as nn
import torch
import torch.nn.functional as F
import itertools
import math

class ArcbId(nn.Module):

    def __init__(self,alpha,beta,gamma,net,M =2,s=64,m=0.75):
        super(ArcbId,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.M = M
        self.bce = nn.BCEWithLogitsLoss()
        self.net =  net
        self.m = m
        self.s = s
        self.gamma = gamma
        self.pd = nn.PairwiseDistance(keepdim=  True)

    def forward(self,outputs,classes,emb,ids):   

        theta = torch.acos(outputs)
        outs = classes*torch.cos(theta+self.m) + (1-classes)*torch.cos(theta-self.m)
        outs.mul_(self.s)
        l = self.gamma * self.bce(outs,classes)


        idd = torch.combinations(ids.flatten(),2)
        cll = torch.combinations(classes.flatten(),2)

        idxx = torch.combinations(torch.arange(0,len(ids)),2)
        
        c1 = torch.logical_and(idd[:,0]!=idd[:,1],cll[:,0]==cll[:,1])# index for same class but diffrent id
        c2 = torch.logical_and(idd[:,0]==idd[:,1],cll[:,0]!=cll[:,1]) # index for difrrent class label but same idreeeeeturn 
        
        c1_mask = torch.nonzero(c1)
        c2_mask = torch.nonzero(c2)
        
        if c1.sum()>0:
            l += self.alpha*( self.pd(emb[idxx[c1_mask,0]],emb[idxx[c1_mask,1]])  ).mean() 
        if c2.sum()>0:
            l += self.beta*(torch.max(torch.zeros_like(c2_mask),self.M-self.pd(emb[idxx[c2_mask,0]],emb[idxx[c2_mask,1]])).mean() )
 
        return l

class ArcB(nn.Module):
    def __init__(self,net, m=0.75,s=64):
        super(ArcB, self).__init__()
        self.m = m 
        self.s = s
        self.net = net
        self.bce = nn.BCEWithLogitsLoss()


    def forward(self,outputs,classes,emb,ids):
        theta = torch.acos(outputs)
        outs = classes*torch.cos(theta+self.m) + (1-classes)*torch.cos(theta-self.m)
        outs.mul_(self.s)

        return self.bce(outs,classes)



class BCEWithLogits(nn.Module):
    def __init__(self):
        super(BCEWithLogits,self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,outputs,classes,emb,ids):
        return self.criterion(outputs, classes)


class IdBce(nn.Module):
    def __init__(self,alpha=0.5,M=0.5):
        super(IdBce,self).__init__()
        self.alpha = alpha
        self.M = M
        self.bce =  nn.BCEWithLogitsLoss()

    def forward(self,outputs,classes,emb,ids):

        l = 0 
        idd = torch.combinations(ids.flatten(),2)
        cll = torch.combinations(classes.flatten(),2)

        idxx = torch.combinations(torch.arange(0,len(ids)),2)
        
        c1 = torch.logical_and(idd[:,0]!=idd[:,1],cll[:,0]==cll[:,1])# index for same class but diffrent id
        c2 = torch.logical_and(idd[:,0]==idd[:,1],cll[:,0]!=cll[:,1]) # index for difrrent class label but same idreeeeeturn 
        
        c1_mask = torch.nonzero(c1)
        c2_mask = torch.nonzero(c2)
        
        if c1.sum()>0:
            l += self.alpha*( self.pd(emb[idxx[c1_mask,0]],emb[idxx[c1_mask,1]])  ).mean() 
        if c2.sum()>0:
            l += self.beta*(torch.max(torch.zeros_like(c2_mask),self.M-self.pd(emb[idxx[c2_mask,0]],emb[idxx[c2_mask,1]])).mean() )

        lbce= self.bce(outputs, classes)

        return lbce + self.alpha*l


""" class ArcbTPId(nn.Module):

    def __init__(self,alpha,beta,gamma,net,M = 0.5,m=0.5):
        super(ArcbTPId,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.M = M
        self.bce = nn.BCEWithLogitsLoss()
        # self.bce = nn.BCELoss()
        self.net =  net
        self.m = m
        self.s = 2.6
        self.gamma = gamma
        
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

        # self.pd2 = nn.PairwiseDistance(keepdim=  False)
        self.cs = nn.CosineSimilarity()
        print("ArcB loss with Person ID based loss")

    def forward(self,outputs,classes,emb,ids):
        # w = list(self.net.parameters())[-1]
        w = self.net.head.weight
        # scale = torch.norm(w)*torch.norm(emb,dim=1)
        scale = torch.norm(w)*torch.norm(emb,dim=1)
        scale = scale.unsqueeze(dim=1)
        
        
        theta = torch.acos(self.cs(emb,w)).unsqueeze(dim=1)
        
        outs = classes*torch.cos(theta+self.m) + (1-classes)*torch.cos(theta-self.m)
        outs = self.s*outs
    
        idd = torch.combinations(ids.reshape(-1),3)
        cll = torch.combinations(classes.reshape(-1),3)
        idxx = torch.combinations(torch.arange(0,len(ids)),3)
        mask = torch.logical_and(cll[:,0]==cll[:,1],idd[:,0]!=idd[:,1]).logical_and(torch.logical_and(cll[:,0]!=cll[:,2],idd[:,0]==idd[:,2]))
        mask = torch.nonzero(mask).flatten()
        anchor = emb[idxx[mask,0]]
        positive = emb[idxx[mask,1]]
        negative = emb[idxx[mask,2]]
        # print(self.triplet_loss(anchor,positive,negative) )
        if len(mask)>0:
            return self.alpha*self.triplet_loss(anchor,positive,negative) + self.beta*self.bce(outs,classes)
        else:
            return self.beta*self.bce(outs,classes)
 """

 
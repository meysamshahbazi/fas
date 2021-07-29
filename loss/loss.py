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

        # for i,j in itertools.combinations(range(len(ids)), 2):
        #     if ids[i]==ids[j] and classes[i]!=classes[j]:
        #         idx1.append([i,j])
        #     if ids[i]!=ids[j] and classes[i]==classes[j]:
        #         idx2.append([i,j])


        # idx1 = torch.Tensor(idx1).type(torch.long)
        # idx2 = torch.Tensor(idx2).type(torch.long)
        # if len(idx1)>0:
        #     l += torch.norm((emb[idx1[:,0]]-emb[idx1[:,1]]),dim=1).mean()
        # if len(idx2)>0:
        #     l += torch.max(torch.zeros(len(idx2),device=emb.device),self.M-torch.norm((emb[idx2[:,0]]-emb[idx2[:,1]]),1,dim=1)).mean()
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
        return self.bce(outputs, classes) + self.alpha*l




class ArcbId(nn.Module):

    def __init__(self,alpha,net,M = 0.5,m=0.5):
        super(ArcbId,self).__init__()
        self.alpha = alpha
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
        outs = classes*scale*torch.cos(theta+self.m) + (1-classes)*scale*torch.cos(theta-self.m)
        l = 0
        # l+=self.bce(outs,classes)

        # idx1 = [] # index for same id but diffrent class label
        # idx2 = [] # index for same class label but diffrent id
        
        emb = F.normalize(emb)

        # for i,j in itertools.combinations(range(len(ids)), 2):
        #     if ids[i]==ids[j] and classes[i]!=classes[j]:
        #         idx1.append([i,j])
        #     if ids[i]!=ids[j] and classes[i]==classes[j]:
        #         idx2.append([i,j])


        # idx1 = torch.Tensor(idx1).type(torch.long)
        # idx2 = torch.Tensor(idx2).type(torch.long)
        # if len(idx1)>0:
        #     l += self.alpha*torch.norm((emb[idx1[:,0]]-emb[idx1[:,1]]),dim=1).mean()
        # if len(idx2)>0:
        #     l += self.alpha*torch.max(torch.zeros(len(idx2),device=emb.device),self.M-torch.norm((emb[idx2[:,0]]-emb[idx2[:,1]]),1,dim=1)).mean()
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

        return self.bce(outs,classes) + self.alpha*l



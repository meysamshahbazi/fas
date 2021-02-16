from torch.utils import data
import cv2
import random
import os
import timeit
import json
import torch

class FASDataset(data.Dataset):
    '''
    this is base class for FAS datasets
    other class must inheritance form this but impliment their own methods for cusotm functionality
    '''
    def __init__(self,root,data_partion,batch_size=1,for_train=True):
        self.root = root
        self.data_partion = data_partion
        self.for_train = for_train # for dev and test this is False
        self.vid_idx = 0
        self.vid_list = []
        self.opened_vid = {}
        self.batch_size = batch_size
        if not os.path.isfile(self.root+self.data_partion+'.json'):
          print("Json doesnt Exist! try to create it and it would take a time!")
          self.crateJsonSummery()
        with open(self.root+self.data_partion+'.json', 'r') as openfile:
            self.datadict = json.load(openfile)  



    def __len__(self):
        return self.datadict[str(len(self.datadict)-1)]['nb_frame_total']
    
    def __getitem__(self,index): 
        if self.for_train:
            self.vid_idx,frame_index = index
            if not self.vid_idx in self.opened_vid.keys():
                if int(len(self.opened_vid)) == self.batch_size:
                    self.opened_vid = {}
                self.opened_vid[self.vid_idx] = []
                cap = cv2.VideoCapture(self.root+self.datadict[str(self.vid_idx)]['name'] )
                while cap.isOpened():
                    ret,frame = cap.read()
                    if ret:
                        self.opened_vid[self.vid_idx].append(frame)
                    else:
                        break
                cap.release()
            x = self.opened_vid[self.vid_idx][frame_index]
            self.opened_vid[self.vid_idx][frame_index] = []
                
        else: # for dev and test
            if index == 0 or index >= self.datadict[str(self.vid_idx)]['nb_frame_total']:
                if index == 0:
                    self.vid_idx = 0
                else: 
                    self.vid_idx += 1
                    
                self.vid_list = []
                cap = cv2.VideoCapture(self.root+self.datadict[str(self.vid_idx)]['name'] )
                while cap.isOpened():
                    ret,frame = cap.read()
                    if ret:
                        self.vid_list.append(frame)
                    else:
                        break
                cap.release()
            
            if self.vid_idx > 0:
                frame_idx = index - self.datadict[str(self.vid_idx-1)]['nb_frame_total']
            else:
                frame_idx = index
            
            x = self.vid_list[frame_idx]
            
        x = x.transpose(2,1,0)
        x = torch.FloatTensor(x)
        y = self.datadict[str(self.vid_idx)]['real_or_spoof']
        y = torch.FloatTensor([y])
        return x,y

    def clear_cache(self):#this must be called after end of each epoch in training
      self.opened_vid = {}

    def crateJsonSummery(self):
        raise NotImplementedError("Subclass must implement abstract method")
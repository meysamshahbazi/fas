from torch.utils import data
import cv2
import random
import os
import timeit
import json
import torch
import math


class FASDataset(data.Dataset):
    '''
    this is base class for FAS datasets
    other class must inheritance form this but impliment their own methods for cusotm functionality
    '''
    def __init__(self,root,data_partion,batch_size=1,for_train=True,shape=(224,224)):
        self.root = root
        self.data_partion = data_partion
        self.for_train = for_train # for dev and test this is False
        self.vid_idx = 0
        self.vid_list = []
        self.opened_vid = {}
        self.batch_size = batch_size
        self.shape = shape
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
                rotate_func = self.get_rotate_func(self.datadict[str(self.vid_idx)]['name'])
                cap = cv2.VideoCapture(self.root+self.datadict[str(self.vid_idx)]['name'] )
                face_locs = self.get_randomed_face_loc(self.vid_idx)
                face_locs_idx = 0
                while cap.isOpened():
                    ret,frame = cap.read()
                    if ret:
                        frame = rotate_func(frame)
                        x1,y1,x2,y2 = face_locs[face_locs_idx % len(face_locs)] # this is becuse some dataset have diffrent frame cnt and lines of .face file!!
                        face_locs_idx +=1
                        frame = frame[y1:y2,x1:x2,:]
                        frame = cv2.resize(frame, dsize=self.shape, interpolation=cv2.INTER_CUBIC)
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
                rotate_func = self.get_rotate_func(self.datadict[str(self.vid_idx)]['name'])
                cap = cv2.VideoCapture(self.root+self.datadict[str(self.vid_idx)]['name'] )
                face_locs = self.get_randomed_face_loc(self.vid_idx)
                face_locs_idx = 0
                while cap.isOpened():
                    ret,frame = cap.read()
                    if ret:
                        frame = rotate_func(frame)
                        x1,y1,x2,y2 = face_locs[face_locs_idx]
                        face_locs_idx +=1
                        frame = frame[y1:y2,x1:x2,:]
                        frame = cv2.resize(frame, dsize=self.shape, interpolation=cv2.INTER_CUBIC)
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
        y = int(self.datadict[str(self.vid_idx)]['real_or_spoof'])
        y = torch.FloatTensor([y])
        id = int(self.datadict[str(self.vid_idx)]['person_id'])
        id = torch.FloatTensor([id])
        return x,y,id

    def clear_cache(self):#this must be called after end of each epoch in training
        self.opened_vid = {}
    def random_crop(self,x1,y1,x2,y2,img_shape):
        scale = max(math.ceil((x2-x1)/self.shape[1]), math.ceil((y2-y1)/self.shape[0]))

        x1 = random.randint(max(0,x2-scale*self.shape[1]),min(x1,max(0,img_shape[1]-scale*self.shape[1])))
        y1 = random.randint(max(0,y2-scale*self.shape[0]),min(y1,max(0,img_shape[0]-scale*self.shape[0])))
        
        x2 = x1 + scale*self.shape[1]
        y2 = y1 + scale*self.shape[0]
        
        return x1,y1,x2,y2

    def get_rotate_func(self,name):
        return lambda x: x

    def crateJsonSummery(self):
        raise NotImplementedError("Subclass must implement abstract method")
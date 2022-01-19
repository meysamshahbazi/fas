from torch.utils import data
from torchvision import transforms
import cv2
import random
import os
import timeit
import json
import torch
import math
from PIL import Image, ImageOps 
from PIL import Image


class FASDataset(data.Dataset):
    '''
    this is base class for FAS datasets
    other class must inheritance form this but impliment their own methods for cusotm functionality
    '''
    def __init__(self,root,data_partion,batch_size=1,for_train=True,shape=(224,224),frame_per_vid=300):
        self.root = root
        self.data_partion = data_partion
        self.for_train = for_train # for dev and test this is False
        self.vid_idx = 0
        self.vid_list = []
        self.opened_vid = {}
        self.batch_size = batch_size
        self.shape = shape
        self.frame_per_vid = frame_per_vid
        if not os.path.isfile(self.root+self.data_partion+'.json'):
          print("Json doesnt Exist! try to create it and it would take a time!")
          self.crateJsonSummery()
        with open(self.root+self.data_partion+'.json', 'r') as openfile:
            self.datadict = json.load(openfile)  
        means,stds = self._get_mean_std()   
        # casia
        # means = [0.4828,0.4291,0.4075]
        # stds = [0.2440,0.2413,0.2505]
        self.transform = transforms.Compose([
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=means,
                                                std=stds),
                            transforms.RandomErasing(scale=(0.02, 0.2))
                            ])
        self.transformd = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=means,
                                                std=stds)
                            ])
        self.rct = transforms.RandomCrop((224,224), padding=0, pad_if_needed=True, fill=0, padding_mode='reflect')
        # self.transform = transforms.Compose([
        #                     transforms.ToTensor()
        # ])

        self.nb_frame_total = 0
        for k in self.datadict.keys():
            self.nb_frame_total+=self.datadict[k]['nb_frame']

    def __len__(self):
        # if self.for_train:
        return self.datadict[str(len(self.datadict)-1)]['nb_frame_total']
        # else:
        #      return len(self.datadict) #TODO: need change 
        # return self.nb_frame_total
    def _get_frames(self,vid_idx,get_face_loc_func):
        rotate_func = self.get_rotate_func(self.datadict[str(vid_idx)]['name'])
        cap = cv2.VideoCapture(self.root+self.datadict[str(vid_idx)]['name'] ,cv2.CAP_FFMPEG)
        # face_locs = self.get_randomed_face_loc(self.vid_idx)
        img_shape = self.datadict[str(vid_idx)]['resolution']
        face_locs = self.get_face_loc(vid_idx) 
        face_locs_idx = 0
        opened_vid = []
        while cap.isOpened():
            ret,frame = cap.read()
            if ret:
                frame = rotate_func(frame)
                x1,y1,x2,y2 = face_locs[face_locs_idx % len(face_locs)] # this is becuse some dataset have diffrent frame cnt and lines of .face file!!
                face_locs_idx +=1
                # x1,y1,x2,y2 = self.get_scaled_face_locs(x1,y1,x2,y2,img_shape)
                x1,y1,x2,y2 = get_face_loc_func(x1,y1,x2,y2,img_shape)
                
                frame = frame[y1:y2,x1:x2,:]
                frame = cv2.resize(frame, dsize=self.shape, interpolation=cv2.INTER_CUBIC)
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
                opened_vid.append(frame)
            else:
                break
        cap.release()
        return opened_vid
    def __getitem__(self,index): 
        if self.for_train:
            self.vid_idx,frame_index = index
            if not self.vid_idx in self.opened_vid.keys():
                if int(len(self.opened_vid)) == self.batch_size:
                    self.opened_vid = {}
                self.opened_vid[self.vid_idx] = self._get_frames(self.vid_idx,self.get_scaled_randomed_face_locs) 
                # self.opened_vid[self.vid_idx] = self.opened_vid[self.vid_idx] #[0:60] #just for siw 

            x = self.opened_vid[self.vid_idx][frame_index%len(self.opened_vid[self.vid_idx])]
            x = self.transform(x)
                
        else: # for dev and test
            if index == 0 or index >= self.datadict[str(self.vid_idx)]['nb_frame_total']:
                if index == 0:
                    self.vid_idx = 0
                else: 
                    self.vid_idx += 1
                    
                self.vid_list = self._get_frames(self.vid_idx,self.get_scaled_face_locs) 
            if self.vid_idx > 0:
                frame_idx = index - self.datadict[str(self.vid_idx-1)]['nb_frame_total']
            else:
                frame_idx = index

            x = self.vid_list[frame_idx]
            x = self.transformd(x)


        y = int(self.datadict[str(self.vid_idx)]['real_or_spoof'])
        y = torch.FloatTensor([y])

        iD = int(self.datadict[str(self.vid_idx)]['person_id'])
        iD = torch.FloatTensor([iD])
        return x,y,iD

    def clear_cache(self):#this must be called after end of each epoch in training
        self.opened_vid = {}
    def get_scaled_randomed_face_locs(self,x1,y1,x2,y2,img_shape,scale=2):
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(img_shape[1],x2)
        y2 = min(img_shape[0],y2)

        xx1 = x1 - (scale-1)/2*(x2-x1)
        xx2 = x2 + (scale-1)/2*(x2-x1)
        yy1 = y1 - (scale-1)/2*(y2-y1)
        yy2 = y2 + (scale-1)/2*(y2-y1)

        xx1 = int(xx1)
        yy1 = int(yy1)
        xx2 = int(xx2)
        yy2 = int(yy2)

        # offset_x = random.randint(2*xx1-x1,x1)
        # offset_y = random.randint(2*yy1-y1,y1)
        offset_x = random.randint(int(- (scale-1)/2*(x2-x1)),int((scale-1)/2*(x2-x1))) # 2 
        offset_y = random.randint(int(- (scale-1)/2*(y2-y1)),int((scale-1)/2*(y2-y1))) # 2

        xx1 = xx1 + offset_x
        xx2 = xx2 + offset_x
        yy1 = yy1 + offset_y
        yy2 = yy2 + offset_y

        xx1 = max(0,xx1)
        yy1 = max(0,yy1)
        xx2 = min(img_shape[1],xx2)
        yy2 = min(img_shape[0],yy2)

        return xx1,yy1,xx2,yy2

    def get_scaled_face_locs(self,x1,y1,x2,y2,img_shape,scale=2):

        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(img_shape[1],x2)
        y2 = min(img_shape[0],y2)

        xx1 = x1 - (scale-1)/2*(x2-x1)
        xx2 = x2 + (scale-1)/2*(x2-x1)
        yy1 = y1 - (scale-1)/2*(y2-y1)
        yy2 = y2 + (scale-1)/2*(y2-y1)
        xx1 = int(xx1)
        yy1 = int(yy1)
        xx2 = int(xx2)
        yy2 = int(yy2)
        xx1 = max(0,xx1)
        yy1 = max(0,yy1)
        xx2 = min(img_shape[1],xx2)
        yy2 = min(img_shape[0],yy2)

        return xx1,yy1,xx2,yy2

    def random_crop(self,x1,y1,x2,y2,img_shape):
        # in some dataset(oulu,casia,rose) bounding box is outter image shape next 4 line will fix this!
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(img_shape[1],x2)
        y2 = min(img_shape[0],y2)

        
        scale = max(math.ceil((x2-x1)/self.shape[1]), math.ceil((y2-y1)/self.shape[0]))

        x1 = random.randint(max(0,x2-scale*self.shape[1]),min(x1,max(0,img_shape[1]-scale*self.shape[1])))
        y1 = random.randint(max(0,y2-scale*self.shape[0]),min(y1,max(0,img_shape[0]-scale*self.shape[0])))
        
        x2 = x1 + scale*self.shape[1]
        y2 = y1 + scale*self.shape[0]
        
        return x1,y1,x2,y2

    def get_rotate_func(self,name):
        return lambda x: x
    def _get_mean_std(self):
        raise NotImplementedError("Subclass must implement abstract method")
    def crateJsonSummery(self):
        raise NotImplementedError("Subclass must implement abstract method")
from torch.utils import data
from torchvision import transforms
import cv2
import random
import os
import timeit
import json
import torch
import math
from PIL import Image
import numpy as np
# from torch_mtcnn import detect_faces


class CasiaFASDImg(data.Dataset):
    def __init__(self,root,data_partion,batch_size=1,shape=(224,224)):
        self.root = root
        self.data_partion = data_partion
        self.batch_size = batch_size
        self.shape = shape
        if not os.path.isfile(self.root+self.data_partion+'.json'):
            print("Json doesnt Exist! try to create it and it would take a time!")
            self.crateJsonSummery()
        with open(self.root+self.data_partion+'.json', 'r') as openfile:
            self.datadict = json.load(openfile) 
            
        self.new_datadict = {}
        index = 0
        for k in self.datadict:
            face_loc_path = self.datadict[k]['name'].split('.')[0]
            face_loc_path = self.root+face_loc_path+'.face'
            with open(face_loc_path, "r") as text_file:
                lines = text_file.readlines()
            for frame_cnt in range(self.datadict[k]['nb_frame']):
                self.new_datadict[index] = {}
                self.new_datadict[index]['name'] = 'train_release-img'+self.datadict[k]['name'].split('train_release')[-1]+'_FRCT_'+str(frame_cnt)+'.png'
                self.new_datadict[index]['real_or_spoof'] = self.datadict[k]['real_or_spoof']
                self.new_datadict[index]['person_id'] = self.datadict[k]['person_id']
                self.new_datadict[index]['resolution'] = self.datadict[k]['resolution']
                l = lines[frame_cnt] 
                x1 = int(l[:-1].split(', ')[1])
                y1 = int(l[:-1].split(', ')[2])
                x2 = int(l[:-1].split(', ')[3])
                y2 = int(l[:-1].split(', ')[4])
                self.new_datadict[index]['face'] = (x1,y1,x2,y2)
                index += 1
                
        self.transform = transforms.Compose([
#                             transforms.ColorJitter(
#                                             brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                            # transforms.RandomAffine(0, translate=(0.1, 0.1)),
                            
#                             transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                            ])
        self.rct = transforms.RandomCrop(224, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
        
    def __len__(self):
        return len(self.new_datadict)
    
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
    def __getitem__(self,index): 
        frame = Image.open(self.root+self.new_datadict[index]['name'])
        x1,y1,x2,y2 = self.new_datadict[index]['face']
        img_shape = self.new_datadict[index]['resolution']
#         if max(x2-x1,y2-y1) < self.shape[0]:
        x1,y1,x2,y2 = self.random_crop(x1,y1,x2,y2,img_shape)
        frame = frame.crop((x1, y1,x2 ,y2))
        if (x2-x1) != self.shape[0] or  (y2-y1) != self.shape[1]:
#             frame = self.rct(frame)
            frame = frame.resize(self.shape)
        x = self.transform(frame)
       
        y = int(self.new_datadict[index]['real_or_spoof'])

        y = torch.FloatTensor([y])
        id = int(self.new_datadict[index]['person_id'])
        id = torch.FloatTensor([id])
        return x,y,id




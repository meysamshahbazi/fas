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
from tqdm import tqdm



with open('/home/user/meysam/OULU-NPU/train.json', 'r') as openfile:
    datadict = json.load(openfile) 

for k in tqdm(datadict.keys()):
    face_locs = []
    img_shape = datadict[k]['resolution']
    face_loc_path = datadict[k]['name'].split('.')[0]
    face_loc_path = '/home/user/meysam/OULU-NPU/'+face_loc_path+'.face'

    with open(face_loc_path, "r") as text_file:
        lines = text_file.readlines()
    for l in lines:
        x1 = int(l[:-1].split(', ')[1])
        y1 = int(l[:-1].split(', ')[2])
        x2 = int(l[:-1].split(', ')[3])
        y2 = int(l[:-1].split(', ')[4])
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(img_shape[1]-1,x2)
        y2 = min(img_shape[0]-1,y2)
        face_locs.append((x1,y1,x2,y2))   
    
    cap = cv2.VideoCapture('/home/user/meysam/OULU-NPU/' + datadict[k]['name'] ,cv2.CAP_FFMPEG)
    face_locs_idx = 0
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            x1,y1,x2,y2 = face_locs[face_locs_idx % len(face_locs)] # this is becuse some dataset have diffrent frame cnt and lines of .face file!!
            face_locs_idx +=1
            frame = frame[y1:y2,x1:x2,:]
            frame = cv2.resize(frame, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('/home/user/meysam/OULU-NPU/train-img/'+datadict[k]['name'].split('/')[1].split('.')[0]+'f_'+str(face_locs_idx)+'.png', frame)
        else:
            break
        
    cap.release()
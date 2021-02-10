from torch.utils import data
import cv2
import random
import os
import timeit
import json
import torch


class ReplayAttack(data.Dataset):
    def __init__(self,root,sub_dir,train_type='train'):
        self.root = root
        self.sub_dir = sub_dir
        self.train_type = train_type # 'train' or 'dev' or 'test'
        with open(self.root+self.sub_dir+'data_summery.json', 'r') as openfile:
            self.datadict = json.load(openfile)  
            
    def __len__(self):
        return self.datadict[str(len(self.datadict)-1)]['nb_frame_total']
    
    def __getitem__(self,index): #TODO: write get item for test and devel 
        vid_idx = 0 # find index of video 
        while self.datadict[str(vid_idx)]['nb_frame_total']<=index:
            vid_idx +=1
        # find index of frame from video number vid_idx
        if vid_idx>0:
            frame_idx = index - self.datadict[str(vid_idx-1)]['nb_frame_total']
        else:
            frame_idx = index
        cap = cv2.VideoCapture(self.root+self.sub_dir+self.datadict[str(vid_idx)]['name'] )
        frame_cnt = 0
        while cap.isOpened():
            ret,frame = cap.read()
            if ret:
                if frame_cnt == frame_idx:
                    x = frame
                    break
                frame_cnt = frame_cnt + 1
            else:
                break
        cap.release()
        x = x.transpose(2,1,0)
        x = torch.FloatTensor(x)
        y = self.datadict[str(vid_idx)]['real_or_spoof']
        y = torch.FloatTensor([y])
        return x,y



def vidToImage(root):
    """
    in replay attack each folder have this form:
    /real
    /attack/fixed
    /attack/hand
    """
    if not os.path.isdir(root[:-1]+'-img'):
        os.mkdir(root[:-1]+'-img')
    if not os.path.isdir(root[:-1]+'-img/real'):    
        os.mkdir(root[:-1]+'-img/real')
    if not os.path.isdir(root[:-1]+'-img/attack'):    
        os.mkdir(root[:-1]+'-img/attack')
    if not os.path.isdir(root[:-1]+'-img/attack/fixed'):    
        os.mkdir(root[:-1]+'-img/attack/fixed')
    if not os.path.isdir(root[:-1]+'-img/attack/hand'):    
        os.mkdir(root[:-1]+'-img/attack/hand')

    path = root+'real/'
    vids = os.listdir(path)
    for v in vids:
        frame_cnt = 0
        cap = cv2.VideoCapture(path+v)
        while cap.isOpened():
            ret,frame = cap.read()
            if ret:
                cv2.imwrite(root[:-1]+'-img/real/'+v[:-4]+'_'+str(frame_cnt)+'.png', frame)
                frame_cnt = frame_cnt + 1
            else:
                break
        cap.release()

    
    
    path = root+'attack/fixed/'
    vids = os.listdir(path)
    for v in vids:
        frame_cnt = 0
        cap = cv2.VideoCapture(path+v)
        while cap.isOpened():
            ret,frame = cap.read()
            if ret:
                cv2.imwrite(root[:-1]+'-img/attack/fixed/'+v[:-4]+'_'+str(frame_cnt)+'.png', frame)
                frame_cnt = frame_cnt + 1
            else:
                break
        cap.release()
        
    path = root+'attack/hand/'
    vids = os.listdir(path)
    for v in vids:
        frame_cnt = 0
        cap = cv2.VideoCapture(path+v)
        while cap.isOpened():
            ret,frame = cap.read()
            if ret:
                cv2.imwrite(root[:-1]+'-img/attack/hand/'+v[:-4]+'_'+str(frame_cnt)+'.png', frame)
                frame_cnt = frame_cnt + 1
            else:
                break
        cap.release()

def crateJsonSummery(root,sub_dir):
    """
    this function creates a dict that describe nessury information of dataset
    and then save that into the json file
    index: name,frame_cnt,label,PAI,face_loc,resolution,person_id,
    """
    my_dict = {}
    vids = []
    vids = os.listdir(root+sub_dir+'real')
    index = 0
    for v in vids:
        my_dict[index] = {}
        my_dict[index]['name'] = 'real/'+v # such that with root+sub_dir+name we can load video
        my_dict[index]['person_id'] = int(v.split('_')[0][-3:])
        my_dict[index]['real_or_spoof'] = 1 # 1 for real and 0 for spoof
        my_dict[index]['PAI'] = None 
        my_dict[index]['lighting'] = v.split('_')[4]
        cap = cv2.VideoCapture(root+sub_dir+my_dict[index]['name'] )
        frame_cnt = 0
        while cap.isOpened():
            ret,frame = cap.read()
            if ret:
                frame_cnt = frame_cnt + 1
                resolution = frame.shape
            else:
                break
        cap.release()
        
        my_dict[index]['nb_frame'] = frame_cnt
        if index == 0:
            my_dict[index]['nb_frame_total'] = frame_cnt
        else:
            my_dict[index]['nb_frame_total'] = frame_cnt + my_dict[index-1]['nb_frame_total'] 
        my_dict[index]['resolution'] = resolution
        
        index = index + 1
    vids = []
    vids = os.listdir(root+sub_dir+'attack/fixed/')

    for v in vids:
        my_dict[index] = {}
        my_dict[index]['name'] = 'attack/fixed/'+v # such that with root+sub_dir+name we can load video
        my_dict[index]['person_id'] = int(v.split('_')[2][-3:])
        my_dict[index]['real_or_spoof'] = 0 # 1 for real and 0 for spoof
        my_dict[index]['PAI'] = v.split('_')[1] 
        my_dict[index]['lighting'] = v.split('_')[6][:-4]
        
        cap = cv2.VideoCapture(root+sub_dir+my_dict[index]['name'] )
        frame_cnt = 0
        while cap.isOpened():
            ret,frame = cap.read()
            if ret:
                frame_cnt = frame_cnt + 1
                resolution = frame.shape
            else:
                break
        cap.release()

        my_dict[index]['nb_frame'] = frame_cnt    
        my_dict[index]['nb_frame_total'] = frame_cnt + my_dict[index-1]['nb_frame_total'] 
        my_dict[index]['resolution'] = resolution
        
        index = index + 1

    vids = []
    vids = os.listdir(root+sub_dir+'attack/hand/')

    for v in vids:
        my_dict[index] = {}
        my_dict[index]['name'] = 'attack/hand/'+v # such that with root+sub_dir+name we can load video
        my_dict[index]['person_id'] = int(v.split('_')[2][-3:])
        my_dict[index]['real_or_spoof'] = 0 # 1 for real and 0 for spoof
        my_dict[index]['PAI'] = v.split('_')[1] 
        my_dict[index]['lighting'] = v.split('_')[6][:-4]
        
        cap = cv2.VideoCapture(root+sub_dir+my_dict[index]['name'] )
        frame_cnt = 0
        while cap.isOpened():
            ret,frame = cap.read()
            if ret:
                frame_cnt = frame_cnt + 1
                resolution = frame.shape
            else:
                break
        cap.release()

        my_dict[index]['nb_frame'] = frame_cnt
        my_dict[index]['nb_frame_total'] = frame_cnt + my_dict[index-1]['nb_frame_total'] 
        my_dict[index]['resolution'] = resolution
        
        index = index + 1
    
        
    
    with open(root+sub_dir+"data_summery.json", "w") as outfile:  
        json.dump(my_dict, outfile) 
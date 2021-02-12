from torch.utils import data
import cv2
import random
import os
import timeit
import json
import torch


class ReplayAttack2(data.Dataset):
    def __init__(self,root,sub_dir,batch_size,for_train=True):
        self.root = root
        self.sub_dir = sub_dir
        self.for_train = for_train # for dev and test this is False
        self.vid_idx = 0
        self.vid_list = []
        self.opened_vid = {}
        self.batch_size = batch_size
        with open(self.root+self.sub_dir+'data_summery.json', 'r') as openfile:
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
                cap = cv2.VideoCapture(self.root+self.sub_dir+self.datadict[str(self.vid_idx)]['name'] )
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
                cap = cv2.VideoCapture(self.root+self.sub_dir+self.datadict[str(self.vid_idx)]['name'] )
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
        
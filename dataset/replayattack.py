from torch.utils import data
from fasdataset import FASDataset
import cv2
import random
import os
import timeit
import json
import torch


class ReplayAttack(FASDataset):
    def __init__(self,root,data_partion,batch_size,for_train=True):
        self.root = root
        self.data_partion = data_partion
        self.for_train = for_train # for dev and test this is False
        self.vid_idx = 0
        self.vid_list = []
        self.opened_vid = {}
        self.batch_size = batch_size
        if not os.path.isfile(self.root+self.data_partion+'/data_summery.json'):
          print("Json doesnt Exist! try to create it and it would take a time!")
          self.crateJsonSummery()
        with open(self.root+self.data_partion+'/data_summery.json', 'r') as openfile:
            self.datadict = json.load(openfile)  
            

    def crateJsonSummery(self):
        """
        this function creates a dict that describe nessury information of dataset
        and then save that into the json file
        index: name,frame_cnt,label,PAI,face_loc,resolution,person_id,
        """
        my_dict = {}
        vids_path = ['/real/','/attack/fixed/','/attack/hand/']
        index = 0
        for vp in vids_path:
            vids = os.listdir(self.root+self.data_partion+vp)
            for v in vids:
                my_dict[index] = {}
                my_dict[index]['name'] = vp+v # such that with root+sub_dir+name we can load video
                                
                if vp == '/real/':  
                    my_dict[index]['person_id'] = int(v.split('_')[0][-3:])
                    my_dict[index]['real_or_spoof'] = 1 # 1 for real and 0 for spoof
                    my_dict[index]['PAI'] = None
                    my_dict[index]['lighting'] = v.split('_')[4]
                else:
                    my_dict[index]['real_or_spoof'] = 0 # 1 for real and 0 for spoof
                    my_dict[index]['PAI'] = v.split('_')[1] 
                    my_dict[index]['person_id'] = int(v.split('_')[2][-3:])
                    my_dict[index]['lighting'] = v.split('_')[6][:-4]
                
                
                cap = cv2.VideoCapture(self.root+self.data_partion+my_dict[index]['name'] )
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
        
        with open(self.root+self.data_partion+"/data_summery.json", "w") as outfile:  
            json.dump(my_dict, outfile) 
            
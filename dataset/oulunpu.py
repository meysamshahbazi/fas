from torch.utils import data
from fasdataset import FASDataset
import cv2
import random
import os
import timeit
import json
import torch
import glob


class OuluNPU(FASDataset):
    def crateJsonSummery(self):
        sub_dir = self.data_partion[0].upper()+self.data_partion[1:]+'_files/'
        vids = glob.glob(self.root+'/'+sub_dir+'*.avi')

        my_dict = {}
        for index,l in enumerate(vids):
            my_dict[index] = {} 
            my_dict[index]['name'] = l.split('/')[-1]
            my_dict[index]['person_id'] = int(l.split('/')[-1][:-4].split('_')[2])
            if int(l.split('/')[-1][:-4].split('_')[3]) ==1:
                my_dict[index]['real_or_spoof'] = 1
                my_dict[index]['PAI'] = None
            elif int(l.split('/')[-1][:-4].split('_')[3]) in [2,3]:
                my_dict[index]['real_or_spoof'] = 0
                my_dict[index]['PAI'] = 'print'
            elif int(l.split('/')[-1][:-4].split('_')[3]) in [4,5]:
                my_dict[index]['real_or_spoof'] = 0
                my_dict[index]['PAI'] = 'replay'
                
            cap = cv2.VideoCapture(self.root+sub_dir+my_dict[index]['name'])
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


        with open(self.root+self.data_partion+".json", "w") as outfile:  
            json.dump(my_dict, outfile) 




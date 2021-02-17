from torch.utils import data
from fasdataset import FASDataset
import cv2
import random
import os
import timeit
import json
import torch

class RoseYoutu(FASDataset):
    
    def crateJsonSummery(self):
        '''
        this will create a json file thath contain usfull info for loading and infering data

        
        '''
        with open(self.root+self.data_partion+'.txt', "r") as text_file:
            lines = text_file.readlines()
        
        my_dict = {}
        for index,l in enumerate(lines):
            my_dict[index] = {} 
            my_dict[index]['name'] = l[:-1]+'.mp4'
            my_dict[index]['person_id'] = int(l[:-1].split('/')[1].split('_')[-2])
            if l[:-1].split('/')[1].split('_')[0] == 'G':
                my_dict[index]['real_or_spoof'] = 1
                my_dict[index]['PAI'] = None
            elif l[:-1].split('/')[1].split('_')[0] in ['Ps','Pq','Mc','Mf','Mu']:
                my_dict[index]['real_or_spoof'] = 0 # 1 for real and 0 for spoof
                my_dict[index]['PAI'] = 'print'
            elif l[:-1].split('/')[1].split('_')[0] in['Vl','Vm']:
                my_dict[index]['real_or_spoof'] = 0 # 1 for real and 0 for spoof
                my_dict[index]['PAI'] = 'replay'

            cap = cv2.VideoCapture(self.root+my_dict[index]['name'])
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
            print(my_dict[index])
        with open(self.root+self.data_partion+".json", "w") as outfile:  
            json.dump(my_dict, outfile) 


# /media/meysam/464C8BC94C8BB26B/ROSE-YOUTU/


if __name__ == "__main__":
    root = '/media/meysam/464C8BC94C8BB26B/ROSE-YOUTU/'
    dataset = RoseYoutu(root,'train')
    dataset = RoseYoutu(root,'test')
    dataset = RoseYoutu(root,'devel')

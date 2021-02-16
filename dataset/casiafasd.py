from torch.utils import data
from fasdataset import FASDataset
import cv2
import random
import os
import timeit
import json
import torch


class CasiaFASD(FASDataset):
    def crateJsonSummery(self):
        # data_partion = 'train' 'test' 'devel'
        #TODO: add PAI, and think about quality!
        
        with open(self.root+self.data_partion+'.txt', "r") as text_file:
            lines = text_file.readlines()
            
        my_dict = {}
        for index,l in enumerate(lines):
            my_dict[index] = {} 
            my_dict[index]['name'] = l[:-1]+'.avi'
            my_dict[index]['person_id'] = l[:-1].split('/')[1]
            if l[:-1].split('/')[2] in ['1','2','HR_1']:
                my_dict[index]['real_or_spoof'] = 1 # 1 for real and 0 for spoof
            else:
                my_dict[index]['real_or_spoof'] = 0 # 1 for real and 0 for spoof
                
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

        with open(self.root+self.data_partion+".json", "w") as outfile:  
            json.dump(my_dict, outfile) 



if __name__ == "__main__":
    root = '/media/meysam/464C8BC94C8BB26B/Casia-FASD/'
    dataset = CasiaFASD(root,'train',4)
    dataset = CasiaFASD(root,'test',4)
    dataset = CasiaFASD(root,'devel',4)
    



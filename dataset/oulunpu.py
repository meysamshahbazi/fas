from torch.utils import data
from dataset.fasdataset import FASDataset
import cv2
import random
import os
import timeit
import json
import torch
import glob
from mtcnn.mtcnn import MTCNN


class OuluNPU(FASDataset):
    def crateJsonSummery(self):
        sub_dir_dict = {'train':'Train_files/','test':'Test_files/','devel':'Dev_files/'} 

        sub_dir = sub_dir_dict[self.data_partion]
        vids = glob.glob(self.root+sub_dir+'*.avi')

        my_dict = {}
        for index,l in enumerate(vids):
            my_dict[index] = {} 
            
            my_dict[index]['name'] = sub_dir+l.split('/')[-1]
    
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

    def createFaceFiles(self):
        sub_dir_dict = {'train':'Train_files/','test':'Test_files/','devel':'Dev_files/'} 

        sub_dir = sub_dir_dict[self.data_partion]
        vids = glob.glob(self.root+sub_dir+'*.avi')
        detector = MTCNN()

        for index,v in enumerate(vids):
            cap = cv2.VideoCapture(v)
            frame_cnt = 0
            my_file = open(v[:-3]+'face','w+')
            while cap.isOpened():
                ret,frame = cap.read()
                if ret:
                    faces = detector.detect_faces(frame)
                    x1,y1,w,h = faces[0]['box']
                    x2 = x1+w
                    y2 = y1+h
                    l  = str(frame_cnt)+', '+str(x1)+', '+str(y1)+', '+str(x2)+', '+str(y2)+'\n'
                    my_file.write(l)
                    frame_cnt = frame_cnt + 1
                    
                else:
                    break
            cap.release()
            my_file.close()




if __name__ == "__main__":
    root = '/media/meysam/B42683242682E6A8/OULU-NPU/'
    dataset = OuluNPU(root,'train')
    dataset.createFaceFiles()
    dataset = OuluNPU(root,'test')
    dataset.createFaceFiles()
    dataset = OuluNPU(root,'devel')
    dataset.createFaceFiles()

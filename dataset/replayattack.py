from torch.utils import data
from dataset.fasdataset import FASDataset
import cv2
import random
import os
import timeit
import json
import torch
import math


class ReplayAttack(FASDataset):
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
                my_dict[index]['name'] = self.data_partion+vp+v # such that with root+name we can load video                
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
                
                
                cap = cv2.VideoCapture(self.root+my_dict[index]['name'] )
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
        
        with open(self.root+self.data_partion+".json", "w") as outfile:  
            json.dump(my_dict, outfile) 
            
    def get_face_loc(self,vid_idx):
        '''
            this function return a list of face location in form of (x1,y1,x2,y2)
        '''
        face_loc_path = self.datadict[str(vid_idx)]['name'].split('.')[0]
        face_loc_path = self.root+'replayattack-face-locations-v2/face-locations/'+face_loc_path+'.face'
        face_locs = []
        with open(face_loc_path, "r") as text_file:
            lines = text_file.readlines()
        for l in lines:
            x1 = int(l[:-1].split(' ')[1])
            y1 = int(l[:-1].split(' ')[2])
            x2 = int(l[:-1].split(' ')[3])+x1
            y2 = int(l[:-1].split(' ')[4])+y1
            face_locs.append((x1,y1,x2,y2))   
        return face_locs


    def get_randomed_face_loc(self,vid_idx):
        '''
            this function return a list of face location in form of (x1,y1,x2,y2)
            with randooized index, so frame will be include a backgraound in order to have same img size
        '''
        img_shape = self.datadict[str(vid_idx)]['resolution']
        face_loc_path = self.datadict[str(vid_idx)]['name'].split('.')[0]
        face_loc_path = self.root+'replayattack-face-locations-v2/face-locations/'+face_loc_path+'.face'
        face_locs = []
        with open(face_loc_path, "r") as text_file:
            lines = text_file.readlines()
        for l in lines:
            x1 = int(l[:-1].split(' ')[1])
            y1 = int(l[:-1].split(' ')[2])
            x2 = int(l[:-1].split(' ')[3])+x1
            y2 = int(l[:-1].split(' ')[4])+y1

            x1,y1,x2,y2 = self.random_crop(x1,y1,x2,y2,img_shape)
            # scale = max(math.ceil((x2-x1)/self.shape[1]), math.ceil((y2-y1)/self.shape[0]))

            # x1 = random.randint(max(0,x2-scale*self.shape[1]),min(x1,img_shape[1]-scale*self.shape[1]))
            # y1 = random.randint(max(0,y2-scale*self.shape[0]),min(y1,img_shape[0]-scale*self.shape[0]))
            
            # x2 = x1 + scale*self.shape[1]
            # y2 = y1 + scale*self.shape[0]
            face_locs.append((x1,y1,x2,y2))   
            
        return face_locs



if __name__ == "__main__":
    root = '/media/meysam/464C8BC94C8BB26B/Replay-Attack/'
    dataset = ReplayAttack(root,'train',4)
    dataset = ReplayAttack(root,'devel',4)
    dataset = ReplayAttack(root,'test',4)

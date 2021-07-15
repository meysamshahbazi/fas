from torch.utils import data
from dataset.fasdataset import FASDataset
import cv2
import random
import os
import timeit
import json
import torch
from tqdm import tqdm

class SiW(FASDataset):


    def crateJsonSummery(self):
        '''
        this will create a json that describe data in human manner!
        '''
        
        with open(self.root+self.data_partion+'.txt', "r") as text_file:
            lines = text_file.readlines()

        my_dict = {}
        for index,l in enumerate(tqdm(lines)):
            my_dict[index] = {} 
            my_dict[index]['name'] = l[:-1]+'.mov'
            my_dict[index]['person_id'] = int(l[:-1].split('/')[-1].split('-')[0])
            if l[:-1].split('/')[-1].split('-')[2] == '1':
                my_dict[index]['real_or_spoof'] = 1
                my_dict[index]['PAI'] = None
            elif l[:-1].split('/')[-1].split('-')[2] == '2':
                my_dict[index]['real_or_spoof'] = 0
                my_dict[index]['PAI'] = 'print'
            elif l[:-1].split('/')[-1].split('-')[2] == '3':
                my_dict[index]['real_or_spoof'] = 0
                my_dict[index]['PAI'] = 'replay'
            else:
                print("ERROR: unexpected type id for:",l)
            
            #TODO: check for existing videos.
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

            #print(my_dict[index])
            #print('-------------------------------------')

        with open(self.root+self.data_partion+".json", "w") as outfile:  
            json.dump(my_dict, outfile) 

    def get_face_loc(self,vid_idx):
        '''
            this function return a list of face location in form of (x1,y1,x2,y2)
        '''
        face_loc_path = self.datadict[str(vid_idx)]['name'].split('.')[0]
        face_loc_path = self.root+face_loc_path+'.face'
        face_locs = []
        with open(face_loc_path, "r") as text_file:
            lines = text_file.readlines()
        for l in lines:
            x1 = int(l[:-1].split(' ')[0])
            y1 = int(l[:-1].split(' ')[1])
            x2 = int(l[:-1].split(' ')[2])
            y2 = int(l[:-1].split(' ')[3])
            face_locs.append((x1,y1,x2,y2))   
        return face_locs

    def get_randomed_face_loc(self,vid_idx):
        '''
            this function return a list of face location in form of (x1,y1,x2,y2)
            with randooized index, so frame will be include a backgraound in order to have same img size
        '''
        img_shape = self.datadict[str(vid_idx)]['resolution']
        face_locs = self.get_face_loc(vid_idx)
        face_locs_random = []
        for x1,y1,x2,y2 in face_locs:
            if (x1,x2,y1,y2)==(0,0,0,0):
                # in this dataset some of frame didnt detect face and have 0,0,0,0
                y2 = img_shape[0]-1
                x2 = img_shape[1]-1
            else:
                x1,y1,x2,y2 = self.random_crop(x1,y1,x2,y2,img_shape)

            face_locs_random.append((x1,y1,x2,y2))  
        # face_loc_path = self.datadict[str(vid_idx)]['name'].split('.')[0]
        # face_loc_path = self.root+face_loc_path+'.face'
        # face_locs = []
        # with open(face_loc_path, "r") as text_file:
        #     lines = text_file.readlines()
        # for l in lines:
        #     x1 = int(l[:-1].split(' ')[0])
        #     y1 = int(l[:-1].split(' ')[1])
        #     x2 = int(l[:-1].split(' ')[2])
        #     y2 = int(l[:-1].split(' ')[3])
        #     x1,y1,x2,y2 = self.random_crop(x1,y1,x2,y2,img_shape)

            # scale = max(math.ceil((x2-x1)/self.shape[1]), math.ceil((y2-y1)/self.shape[0]))

            # x1 = random.randint(max(0,x2-scale*self.shape[1]),min(x1,img_shape[1]-scale*self.shape[1]))
            # y1 = random.randint(max(0,y2-scale*self.shape[0]),min(y1,img_shape[0]-scale*self.shape[0]))
            
            # x2 = x1 + scale*self.shape[1]
            # y2 = y1 + scale*self.shape[0]
            #face_locs.append((x1,y1,x2,y2))   
            
        return face_locs




if __name__ == "__main__":
    root = '/media/meysam/901292F51292E010/SiW/SiW_release/'
    dataset = SiW(root,'train')
    dataset = SiW(root,'test')
    dataset = SiW(root,'devel')


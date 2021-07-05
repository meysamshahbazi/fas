from torch.utils import data
from .fasdataset import FASDataset
import cv2
import random
import os
import timeit
import json
import torch
import glob
#from mtcnn.mtcnn import MTCNN
from torch_mtcnn import detect_faces
from PIL import Image
from tqdm import tqdm


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

        for index,v in enumerate(tqdm(vids)):
            if not os.path.isfile(v[:-3]+'face'):
                cap = cv2.VideoCapture(v)
                frame_cnt = 0
                my_file = open(v[:-3]+'face','w+')
                while cap.isOpened():
                    ret,frame = cap.read()
                    if ret:
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        faces, _ = detect_faces(pil_img)

                        if (len(faces)>0):
                            faces = list(map(int, faces[0]))
                            x1 = faces[0]
                            y1 = faces[1]
                            x2 = faces[2]
                            y2 = faces[3]
                        else:
                            if frame_cnt==0:
                                x1=0
                                y1=0
                                x2=frame.shape[1]-1
                                y2=frame.shape[0]-1
                            print(v)#use preveus detected face!
                            print(frame_cnt)
                        l  = str(frame_cnt)+', '+str(x1)+', '+str(y1)+', '+str(x2)+', '+str(y2)+'\n'
                        my_file.write(l)
                        frame_cnt = frame_cnt + 1
                        
                    else:
                        break
                cap.release()
                my_file.close()

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
            x1 = int(l[:-1].split(', ')[1])
            y1 = int(l[:-1].split(', ')[2])
            x2 = int(l[:-1].split(', ')[3])
            y2 = int(l[:-1].split(', ')[4])
            face_locs.append((x1,y1,x2,y2))   
        return face_locs

    def get_randomed_face_loc(self,vid_idx):
        '''
            this function return a list of face location in form of (x1,y1,x2,y2)
            with randooized index, so frame will be include a backgraound in order to have same img size
        '''
        img_shape = self.datadict[str(vid_idx)]['resolution']
        face_loc_path = self.datadict[str(vid_idx)]['name'].split('.')[0]
        face_loc_path = self.root+face_loc_path+'.face'
        face_locs = []
        with open(face_loc_path, "r") as text_file:
            lines = text_file.readlines()
        for l in lines:
            x1 = int(l[:-1].split(', ')[1])
            y1 = int(l[:-1].split(', ')[2])
            x2 = int(l[:-1].split(', ')[3])
            y2 = int(l[:-1].split(', ')[4])

            x1 = random.randint(max(0,x2-self.shape[1]),min(x1,img_shape[1]-self.shape[1]))
            y1 = random.randint(max(0,y2-self.shape[0]),min(y1,img_shape[0]-self.shape[0]))
            
            x2 = x1 + self.shape[1]
            y2 = y1 + self.shape[0]
            face_locs.append((x1,y1,x2,y2))   
            
        return face_locs


if __name__ == "__main__":
    root = '/media/meysam/B42683242682E6A8/OULU-NPU/'
    dataset = OuluNPU(root,'train')
    dataset.createFaceFiles()
    dataset = OuluNPU(root,'test')
    dataset.createFaceFiles()
    dataset = OuluNPU(root,'devel')
    dataset.createFaceFiles()

from torch.utils import data
from dataset.fasdataset import FASDataset
import cv2
import random
import os
import timeit
import json
import torch
import math
# from torch_mtcnn import detect_faces
from tqdm import tqdm
from PIL import Image


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
    def createFaceFiles(self):
        # with open(self.root+self.data_partion+'.txt', "r") as text_file:
        #     lines = text_file.readlines()

        vids = [self.root+l['name'] for _,l in self.datadict.items()]

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
        face_loc_path = self.root+'replayattack-face-locations-v2/face-locations/'+face_loc_path+'.face'
        face_locs = []
        img_shape = self.datadict[str(vid_idx)]['resolution']
        with open(face_loc_path, "r") as text_file:
            lines = text_file.readlines()
        for l in lines:
            x1 = int(l[:-1].split(' ')[1])
            y1 = int(l[:-1].split(' ')[2])
            x2 = int(l[:-1].split(' ')[3])+x1
            y2 = int(l[:-1].split(' ')[4])+y1
            x1 = max(0,x1)
            y1 = max(0,y1)
            x2 = min(img_shape[1]-1,x2)
            y2 = min(img_shape[0]-1,y2)
            if (x1,x2,y1,y2)==(0,0,0,0):
                # in this dataset some of frame didnt detect face and have 0,0,0,0
                y2 = img_shape[0]-1
                x2 = img_shape[1]-1
            face_locs.append((x1,y1,x2,y2))   
        return face_locs
    def get_face_loc1(self,vid_idx):
        '''
            this function return a list of face location in form of (x1,y1,x2,y2)
        '''
        # print('new face loc')
        face_loc_path = self.datadict[str(vid_idx)]['name'].split('.')[0]
        face_loc_path = self.root+face_loc_path+'.face'
        face_locs = []
        img_shape = self.datadict[str(vid_idx)]['resolution']
        with open(face_loc_path, "r") as text_file:
            lines = text_file.readlines()
        for l in lines:
            x1 = int(l[:-1].split(', ')[1])
            y1 = int(l[:-1].split(', ')[2])
            x2 = int(l[:-1].split(', ')[3])
            y2 = int(l[:-1].split(', ')[4])
            x1 = max(0,x1)
            y1 = max(0,y1)
            x2 = min(img_shape[1]-1,x2)
            y2 = min(img_shape[0]-1,y2)
            face_locs.append((x1,y1,x2,y2))   
        return face_locs
    def _get_mean_std(self):
        means = [0.5557,0.4941,0.4673]
        stds = [0.2910,0.2873,0.2998]
        return means,stds

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
        # face_loc_path = self.root+'replayattack-face-locations-v2/face-locations/'+face_loc_path+'.face'
        # face_locs = []
        # with open(face_loc_path, "r") as text_file:
        #     lines = text_file.readlines()

                    
        # for i,l in enumerate(lines):
        #     x1 = int(l[:-1].split(' ')[1])
        #     y1 = int(l[:-1].split(' ')[2])
        #     x2 = int(l[:-1].split(' ')[3])+x1
        #     y2 = int(l[:-1].split(' ')[4])+y1
        #     if (x1,x2,y1,y2)==(0,0,0,0):
        #         if i==0:
        #             y2 = img_shape[0]-1
        #             x2 = img_shape[1]-1
        #         else:
        #             ll = lines[i-1]
                
        #     x1,y1,x2,y2 = self.random_crop(x1,y1,x2,y2,img_shape)
        #     # scale = max(math.ceil((x2-x1)/self.shape[1]), math.ceil((y2-y1)/self.shape[0]))

        #     # x1 = random.randint(max(0,x2-scale*self.shape[1]),min(x1,img_shape[1]-scale*self.shape[1]))
        #     # y1 = random.randint(max(0,y2-scale*self.shape[0]),min(y1,img_shape[0]-scale*self.shape[0]))
            
        #     # x2 = x1 + scale*self.shape[1]
        #     # y2 = y1 + scale*self.shape[0]
        #     face_locs.append((x1,y1,x2,y2))   
            
        return face_locs_random



if __name__ == "__main__":
    root = '/media/meysam/464C8BC94C8BB26B/Replay-Attack/'
    # dataset = ReplayAttack(root,'train',4)
    # dataset = ReplayAttack(root,'devel',4)
    dataset = ReplayAttack(root,'test',4)
    dataset.createFaceFiles()

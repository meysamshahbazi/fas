from torch.utils import data
from dataset.fasdataset import FASDataset
import cv2
import random
import os
import timeit
import json
import torch
from torch_mtcnn import detect_faces
from tqdm import tqdm
from PIL import Image

grp1 = ['G_NT_5s','G_T_5s','G_NT_IP','G_T_IP','Mc_NT_5s','Mc_T_5s','Mc_NT_IP','Mc_T_IP', \
            'Mf_NT_5s','Mf_T_5s','Mf_NT_IP','Mf_T_IP','Mu_NT_5s','Mu_T_5s','Mu_NT_IP','Mu_T_IP',\
            'Pq_NT_5s','Pq_T_5s','Pq_NT_IP','Pq_T_IP','Ps_NT_5s','Ps_T_5s','Ps_NT_IP','Ps_T_IP']
        
grp2 = ['G_NT_HS','G_T_HS','G_NT_HW','G_T_HW','G_NT_ZTE','G_T_ZTE','Mc_NT_HS','Mc_T_HS', \
        'Mc_NT_HW','Mc_T_HW','Mc_NT_ZTE','Mc_T_ZTE','Mf_NT_HS','Mf_T_HS','Mf_NT_HW','Mf_T_HW',\
    'Mf_NT_ZTE','Mf_T_ZTE','Mu_NT_HS','Mu_T_HS','Mu_NT_HW','Mu_T_HW','Mu_NT_ZTE','Mu_T_ZTE',\
        'Pq_NT_HS','Pq_T_HS','Pq_NT_HW','Pq_T_HW','Pq_NT_ZTE','Pq_T_ZTE',\
        'Ps_NT_HS','Ps_T_HS','Ps_NT_HW','Ps_T_HW','Ps_NT_ZTE','Ps_T_ZTE'
    ]
grp3 = ['Vl_NT_5s','Vl_T_5s','Vl_NT_IP','Vl_T_IP','Vm_NT_5s','Vm_T_5s','Vm_NT_IP','Vm_T_IP']
grp4 = ['Vl_NT_HS','Vl_T_HS','Vl_NT_HW','Vl_T_HW','Vl_NT_ZTE','Vl_T_ZTE',\
    'Vm_NT_HS','Vm_T_HS','Vm_NT_HW','Vm_T_HW','Vm_NT_ZTE','Vm_T_ZTE']

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
            #print(my_dict[index])
        with open(self.root+self.data_partion+".json", "w") as outfile:  
            json.dump(my_dict, outfile) 

    def createFaceFiles(self):
        with open(self.root+self.data_partion+'.txt', "r") as text_file:
            lines = text_file.readlines()

        vids = [self.root+l[:-1]+'.mp4' for l in lines]

        for index,v in enumerate(tqdm(vids)):
            if not os.path.isfile(v[:-3]+'face'):
                rotate_func = self.get_rotate_func(v)
                cap = cv2.VideoCapture(v)
                frame_cnt = 0
                my_file = open(v[:-3]+'face','w+')
                while cap.isOpened():
                    ret,frame = cap.read()
                    if ret:
                        frame = rotate_func(frame)
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

        
    def get_rotate_func(self,name):
        parts = name.split('/')[-1].split('_')
        first3 = parts[0] + '_' + parts[1] + '_' + parts[2]
        
        if first3 in grp1:
            return lambda x: x.transpose(1,0,2)[::-1,:,:] 
        elif first3 in grp2:
            return lambda x: x.transpose(1,0,2)[:,::-1,:]
        elif first3 in grp3:
            return lambda x: x
        elif first3 in grp4:
            return lambda x: x[::-1,::-1,:]
        else:
            print("ERROR! ",name)
            return lambda x: x 


    def get_randomed_face_loc(self,vid_idx):
        '''
            this function return a list of face location in form of (x1,y1,x2,y2)
            with randooized index, so frame will be include a backgraound in order to have same img size
        '''
        img_shape = self.datadict[str(vid_idx)]['resolution']
        # change shape ...
        parts = self.datadict[str(vid_idx)]['name'].split('/')[-1].split('_')
        first3 = parts[0] + '_' + parts[1] + '_' + parts[2]
        if (first3 in grp1)or (first3 in grp2):
            img_shape = [img_shape[1],img_shape[0],img_shape[2]]

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
            x1,y1,x2,y2 = self.random_crop(x1,y1,x2,y2,img_shape)

            # scale = max(math.ceil((x2-x1)/self.shape[1]), math.ceil((y2-y1)/self.shape[0]))

            # x1 = random.randint(max(0,x2-scale*self.shape[1]),min(x1,img_shape[1]-scale*self.shape[1]))
            # y1 = random.randint(max(0,y2-scale*self.shape[0]),min(y1,img_shape[0]-scale*self.shape[0]))
            
            # x2 = x1 + scale*self.shape[1]
            # y2 = y1 + scale*self.shape[0]
            
            face_locs.append((x1,y1,x2,y2))   
            
        return face_locs    
    

# /media/meysam/464C8BC94C8BB26B/ROSE-YOUTU/


if __name__ == "__main__":
    root = '/media/meysam/464C8BC94C8BB26B/ROSE-YOUTU/'
    dataset = RoseYoutu(root,'train')
    dataset = RoseYoutu(root,'test')
    dataset = RoseYoutu(root,'devel')


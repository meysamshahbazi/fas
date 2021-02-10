from torch.utils import data
import cv2
import random
import os
import timeit
import json
import torch
import glob


def crateJsonSummery(root):
    subdir = 'metas/intra_test/'
    with open(root+subdir+'train_label.json', 'r') as openfile:
        datadict1 = json.load(openfile)
    with open(root+subdir+'test_label.json', 'r') as openfile:
        datadict2 = json.load(openfile)
    
    all_in_train_folder = glob.glob(root+'Data/train/*')
    all_in_train_folder.sort()

    train_sub = all_in_train_folder[:6*1024]
    dev_sub = all_in_train_folder[6*1024:]

    train_img = []
    for s in train_sub:
        train_img += glob.glob(s+'/*/*.jpg')+glob.glob(s+'/*/*.png')
        
    dev_img = []
    for s in dev_sub:
        dev_img += glob.glob(s+'/*/*.jpg')+glob.glob(s+'/*/*.png')

    my_dict = {}
    for index,l in enumerate(train_img):
        my_dict[index] = {} 
        my_dict[index]['name'] = l[len(root):]
        my_dict[index]['person_id'] = int(l.split('/')[-3])
        if l.split('/')[-2] =='live':
            my_dict[index]['real_or_spoof'] = 1
            my_dict[index]['PAI'] = None
        elif l.split('/')[-2] =='spoof':
            my_dict[index]['real_or_spoof'] = 0
            if datadict1[my_dict[index]['name']][40] in [1,2,3,4,5,6]:
                my_dict[index]['PAI'] = 'print'
            elif datadict1[my_dict[index]['name']][40] in [7,8,9]:
                my_dict[index]['PAI'] = 'replay'
            elif datadict1[my_dict[index]['name']][40] == 10:
                my_dict[index]['PAI'] = 'mask'
            else:
                print("ERROR:could not find PAI for ",l)
        else:
            print("ERROR:could not find live or spoof label for:",l)
        #TODO: add more detail
        img=cv2.imread(root+my_dict[index]['name'])
        my_dict[index]['resolution'] = img.shape
        with open(root+my_dict[index]['name'][:-4]+'_BB.txt', "r") as text_file:
            lines = text_file.readlines()
        x = lines[0].split(' ')[0]
        y = lines[0].split(' ')[1]
        w = lines[0].split(' ')[2]
        h = lines[0].split(' ')[3]
        my_dict[index]['face_loc'] =(x,y,w,h)

    with open(root+"train.json", "w") as outfile:  
        json.dump(my_dict, outfile) 
    my_dict = {}
    for index,l in enumerate(dev_img):
        my_dict[index] = {} 
        my_dict[index]['name'] = l[len(root):]
        my_dict[index]['person_id'] = int(l.split('/')[-3])
        if l.split('/')[-2] =='live':
            my_dict[index]['real_or_spoof'] = 1
            my_dict[index]['PAI'] = None
        elif l.split('/')[-2] =='spoof':
            my_dict[index]['real_or_spoof'] = 0
            if datadict1[my_dict[index]['name']][40] in [1,2,3,4,5,6]:
                my_dict[index]['PAI'] = 'print'
            elif datadict1[my_dict[index]['name']][40] in [7,8,9]:
                my_dict[index]['PAI'] = 'replay'
            elif datadict1[my_dict[index]['name']][40] == 10:
                my_dict[index]['PAI'] = 'mask'
            else:
                print("ERROR:could not find PAI for ",l)
        else:
            print("ERROR:could not find live or spoof label for:",l)
        #TODO: add more detail
        img=cv2.imread(root+my_dict[index]['name'])
        my_dict[index]['resolution'] = img.shape
        with open(root+my_dict[index]['name'][:-4]+'_BB.txt', "r") as text_file:
            lines = text_file.readlines()
        x = lines[0].split(' ')[0]
        y = lines[0].split(' ')[1]
        w = lines[0].split(' ')[2]
        h = lines[0].split(' ')[3]
        my_dict[index]['face_loc'] =(x,y,w,h)

    with open(root+"dev.json", "w") as outfile:  
        json.dump(my_dict, outfile) 

    test_img = glob.glob(root+'Data/test/*/*/*.png')+glob.glob(root+'Data/test/*/*/*.jpg')

    my_dict = {}
    for index,l in enumerate(test_img):
        my_dict[index] = {} 
        my_dict[index]['name'] = l[len(root):]
        my_dict[index]['person_id'] = int(l.split('/')[-3])
        if l.split('/')[-2] =='live':
            my_dict[index]['real_or_spoof'] = 1
            my_dict[index]['PAI'] = None
        elif l.split('/')[-2] =='spoof':
            my_dict[index]['real_or_spoof'] = 0
            if datadict2[my_dict[index]['name']][40] in [1,2,3,4,5,6]:
                my_dict[index]['PAI'] = 'print'
            elif datadict2[my_dict[index]['name']][40] in [7,8,9]:
                my_dict[index]['PAI'] = 'replay'
            elif datadict2[my_dict[index]['name']][40] == 10:
                my_dict[index]['PAI'] = 'mask'
            else:
                print("ERROR:could not find PAI for ",l)
        else:
            print("ERROR:could not find live or spoof label for:",l)
        #TODO: add more detail
        img=cv2.imread(root+my_dict[index]['name'])
        my_dict[index]['resolution'] = img.shape
        with open(root+my_dict[index]['name'][:-4]+'_BB.txt', "r") as text_file:
            lines = text_file.readlines()
        x = lines[0].split(' ')[0]
        y = lines[0].split(' ')[1]
        w = lines[0].split(' ')[2]
        h = lines[0].split(' ')[3]
        my_dict[index]['face_loc'] =(x,y,w,h)

    with open(root+"test.json", "w") as outfile:  
        json.dump(my_dict, outfile) 
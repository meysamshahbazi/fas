import argparse
import torch.nn as nn
import torch
from protocol.protocol import calcEER
from dataset.sampler import TrainBatchSampler
from dataset.casiafasd import CasiaFASD
from dataset.msumfsd import MsuFsd
from dataset.oulunpu import OuluNPU
from dataset.roseyoutu import RoseYoutu
from dataset.siw import SiW
from model.cnn import CNN
from torch.utils import data
import matplotlib.pyplot as plt
import os




#root = '/media/meysam/B42683242682E6A8/OULU-NPU/'
#dataset = OuluNPU(root,'train')
#dataset.createFaceFiles()
#dataset = OuluNPU(root,'test')
#dataset.createFaceFiles()
#dataset = OuluNPU(root,'devel')
#dataset.createFaceFiles()

#root = '/media/meysam/464C8BC94C8BB26B/Casia-FASD/'
#dataset = CasiaFASD(root,'train',4)
#dataset.createFaceFiles()
#dataset = CasiaFASD(root,'test',4)
#dataset.createFaceFiles()
#dataset = CasiaFASD(root,'devel',4)
#dataset.createFaceFiles()

root = '/media/meysam/464C8BC94C8BB26B/ROSE-YOUTU/'
#dataset = RoseYoutu(root,'train')
#dataset.createFaceFiles()

dataset = RoseYoutu(root,'devel')
dataset.createFaceFiles()

dataset = RoseYoutu(root,'test')
dataset.createFaceFiles()

root = '/media/meysam/B42683242682E6A8/OULU-NPU/'

dataset = OuluNPU(root,'test')
dataset.createFaceFiles()



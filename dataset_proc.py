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
from model.alexnet import AlexNet,AlexNetLite
from model.cnn import CNN
from loss.loss import BCEWithLogits,ArcB,IdBce,ArcbId
from torch.utils import data
import matplotlib.pyplot as plt
import os
import timeit
import numpy as np



root = '/media/meysam/B42683242682E6A8/OULU-NPU/'
#dataset = OuluNPU(root,'train')
#dataset.createFaceFiles()
#dataset = OuluNPU(root,'test')
#dataset.createFaceFiles()
dataset = OuluNPU(root,'devel')
dataset.createFaceFiles()



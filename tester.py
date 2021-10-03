import argparse
from model.resnext import resnext50_32x4d
import torch.nn as nn
import torch
from protocol.protocol import calcEER,calcHTER
from dataset.sampler import TrainBatchSampler
from dataset.replayattack import ReplayAttack
from dataset.casiafasd import CasiaFASD
from dataset.msumfsd import MsuFsd
from dataset.oulunpu import OuluNPU
from dataset.roseyoutu import RoseYoutu
from dataset.siw import SiW
from model import *
from loss.loss import BCEWithLogits,ArcB,IdBce,ArcbId
from torch.utils import data
import matplotlib.pyplot as plt
import os
import timeit
import numpy as np
from tqdm import tqdm


def proc_args():
    parser = argparse.ArgumentParser(description='base argument of test loop')
    parser.add_argument(
    '--dataset',
    type = str,
    required=True,
    help='the dataset for test can be following:\n \
    casia\n \
    celeb\n \
    msu\n \
    oulu\n \
    replay\n \
    rose\n \
    siw\n '
    )
    parser.add_argument(
    '--backbone',
    type = str,
    required=True,
    help='the backbone for train and dev which can be following:\n \
    resnext50_32x4d\n \
    resnext101_32x8d\n '
    )
    #threshold
    parser.add_argument(
    '--threshold',
    required=True,
    type = float,
    help='threshold which we have EER'
    )
    parser.add_argument(
    '--use_lbp',
    type = int,
    default=0,
    help='whether or not using lbp befor backbone.\n '
    )
    parser.add_argument(
    '--tr_on',
    type = str,
    required=True,
    help='the model trained on .\n '
    )
    parser.add_argument(
    '--weights',
    type = str,
    required=True,
    help='weights of trained model.\n '
    )

    parser.add_argument(
    '--lbp_ch',
    type = int,
    default=3,
    help='nubmer of channel outputs for lbp operator (default: 8).\n '
    )
    
    
    parser.add_argument(
    '--emb_size',
    default=512,
    type = int,
    help='size of embeddeing space (default: 512)'
    )
    #
    parser.add_argument(
    '--input_size',
    default=224,
    type = int,
    help='size of input image to model (default: 112)'
    )
    
    parser.add_argument(
    '--test_batch_size',
    default=256,
    type = int,
    help='batch size for test (default: 256)'
    )
    parser.add_argument(
    '--dbg',
    default=False,
    type = bool,
    help='Print Debug info (default: False)'
    )

    
    parser.add_argument(
    '--path',
    default='./',
    type = str,
    help='path (default: 2)'
    )
    cfg = parser.parse_args()

    return cfg


def get_dataset(cfg):
    shape = (cfg.input_size,cfg.input_size)
    data_partion = 'test'
    if cfg.dataset == 'casia':
        root = '/media/meysam/464C8BC94C8BB26B/Casia-FASD/'
        test_dataset = CasiaFASD(root,data_partion,cfg.test_batch_size,for_train=False,shape=shape)
    elif cfg.dataset == 'msu':
        root = '/media/meysam/464C8BC94C8BB26B/MSU-MFSD/'
        #root = '/home/meysam/Desktop/MSU-MFSD/MSU-MFSD-Publish/'
        test_dataset = MsuFsd(root,data_partion,cfg.test_batch_size,for_train=False,shape=shape)
    elif cfg.dataset == 'oulu':
        root = '/media/meysam/B42683242682E6A8/OULU-NPU/'
        test_dataset = OuluNPU(root,data_partion,cfg.test_batch_size,for_train=False,shape=shape)
    elif cfg.dataset == 'replay':
        root = '/media/meysam/464C8BC94C8BB26B/Replay-Attack/' 
        # root = '/home/meysam/Desktop/Replay-Attack/'
        # root = '/content/replayattack/'
        test_dataset = ReplayAttack(root,data_partion,cfg.test_batch_size,for_train=False,shape=shape)
    elif cfg.dataset == 'rose':
        root = '/media/meysam/464C8BC94C8BB26B/ROSE-YOUTU/'
        test_dataset = RoseYoutu(root,data_partion,cfg.test_batch_size,for_train=False,shape=shape)
    elif cfg.dataset == 'siw':
        root = '/media/meysam/901292F51292E010/SiW/SiW_release/'
        test_dataset = SiW(root,data_partion,cfg.test_batch_size,for_train=False,shape=shape)
    else:
        print("Error: unsuported datset!!")
    
    
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=cfg.test_batch_size, 
                               shuffle= False, sampler = None, batch_sampler = None,
                               num_workers= 0, collate_fn= None, pin_memory = False, 
                               drop_last = False, timeout= 0, worker_init_fn = None,
                               multiprocessing_context=None, generator=None, prefetch_factor = 2,
                               persistent_workers = False)

    return test_loader

# def test(net,criterion,dev_loader,device,epoch,path):
#     loss = 0.0
#     iterations = 0
#     sigmoid = nn.Sigmoid()
#     lbl = []
#     pred = []
#     net.eval() # Put the network into evaluate mode

#     for i, (items, classes,ids) in enumerate(tqdm(dev_loader)):
#         torch.cuda.empty_cache()
#         # Convert torch tensor to Variable
#         with torch.no_grad():
#             items = items.to(device)
#             classes = classes.to(device)
#             ids = ids.to(device)
#             outputs,emb = net(items)      # Do the forward pass
#             loss += criterion(outputs, classes,emb,ids).item() # Calculate the loss
#             outputs = sigmoid(outputs) # use sigmoid for infering
#         # Record the all labels of dataset and prediction of model for later use!
#         lbl += classes.cpu().flatten().tolist()    
#         pred += outputs.detach().cpu().flatten().tolist()

#         iterations += 1
#     # Record the validation loss
#     dev_loss.append(loss/iterations)
#     FAR,FRR,threshold = calcEER(lbl,pred,epoch,path,plot=True,precision=0.01)



def main():
    
    cfg = proc_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    test_loader = get_dataset(cfg)

    net = Model(cfg)
    net.load_state_dict(torch.load(cfg.weights))

    net.to(device)

    path = 'tr-on_'+cfg.tr_on+'_test-on_'+cfg.dataset
    os.mkdir('outputs/'+path)
    os.mkdir('outputs/'+path+'/eer_figs')
    sigmoid = nn.Sigmoid()
    lbl = []
    pred = []
    net.eval() # Put the network into evaluate mode

    for i, (items, classes,ids) in enumerate(tqdm(test_loader)):
        torch.cuda.empty_cache()
        # Convert torch tensor to Variable
        with torch.no_grad():
            items = items.to(device)
            classes = classes.to(device)
            ids = ids.to(device)
            outputs,emb = net(items)      # Do the forward pass
            outputs = sigmoid(outputs) # use sigmoid for infering
        # Record the all labels of dataset and prediction of model for later use!
        lbl += classes.cpu().flatten().tolist()    
        pred += outputs.detach().cpu().flatten().tolist()
        # print(len(pred))
        # print(min(lbl))


    FAR,FRR,threshold = calcEER(lbl,pred,0,path,plot=True,precision=0.01)
    HTER = calcHTER(lbl,pred,cfg.threshold,path)
    # AUC = calcAUC(lbl,pred,precision=0.01)
     




if __name__ == "__main__":
    train_loss = []
    dev_loss = []
    main()





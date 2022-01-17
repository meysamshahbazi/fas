from proc_args import proc_args
import torch.nn as nn
import torch
from protocol.protocol import calcEER
from dataset.sampler import TrainBatchSampler,TrainBatchSampler2,TrainBatchSampler3,TrainBatchSampler4
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
from proc_args import proc_args




def train(net,criterion,optimizer,train_loader,device,epoch,path):
    global train_loss
    iter_loss = 0.0
    iterations = 0
    
    net.train()                   # Put the network into training mode
    optimizer.zero_grad()     # Clear off the gradients from any past operation
    for i, (items, classes,ids) in enumerate(tqdm(train_loader)):
        # If we have GPU, shift the data to GPU
        items = items.to(device)
        classes = classes.to(device)
        ids = ids.to(device)

        optimizer.zero_grad()     # Clear off the gradients from any past operation

        outputs,emb = net(items)      # Do the forward pass

        loss = criterion(outputs, classes,emb,ids) # Calculate the loss

        iter_loss += loss.item() # Accumulate the loss

        loss.backward()           # Calculate the gradients with help of back propagation

        optimizer.step()          # Ask the optimizer to adjust the parameters based on the gradients

        iterations += 1
    # Record the training loss
    train_loss.append(iter_loss/iterations)



def dev(net,criterion,dev_loader,device,epoch,path):
    global dev_loss
    loss = 0.0
    iterations = 0
    sigmoid = nn.Sigmoid()
    lbl = []
    pred = []
    net.eval() # Put the network into evaluate mode

    for i, (items, classes,ids) in enumerate(tqdm(dev_loader)):
        torch.cuda.empty_cache()
        # Convert torch tensor to Variable
        with torch.no_grad():
            items = items.to(device)
            classes = classes.to(device)
            ids = ids.to(device)
            outputs,emb = net(items)      # Do the forward pass
            loss += criterion(outputs, classes,emb,ids).item() # Calculate the loss
            outputs = sigmoid(outputs) # use sigmoid for infering
        # Record the all labels of dataset and prediction of model for later use!
        lbl += classes.cpu().flatten().tolist()    
        pred += outputs.detach().cpu().flatten().tolist()

        iterations += 1
    # Record the validation loss
    dev_loss.append(loss/iterations)
    FAR,FRR,threshold = calcEER(lbl,pred,epoch,path,plot=True,precision=0.01)





def get_dataset(cfg):
    shape = (cfg.input_size,cfg.input_size)
    if cfg.dataset == 'casia':
        root = '/media/meysam/464C8BC94C8BB26B/Casia-FASD/'
        data_partion = 'train'
        train_dataset = CasiaFASD(root,data_partion,cfg.train_batch_size,for_train=True,shape=shape)
        data_partion = 'devel'
        dev_dataset = CasiaFASD(root,data_partion,cfg.devel_batch_size,for_train=False,shape=shape)
    elif cfg.dataset == 'msu':
        root = '/media/meysam/464C8BC94C8BB26B/MSU-MFSD/'
        #root = '/home/meysam/Desktop/MSU-MFSD/MSU-MFSD-Publish/'
        data_partion = 'train'
        train_dataset = MsuFsd(root,data_partion,cfg.train_batch_size,for_train=True,shape=shape)
        data_partion = 'devel'
        dev_dataset = MsuFsd(root,data_partion,cfg.devel_batch_size,for_train=False,shape=shape)
    elif cfg.dataset == 'oulu':
        root = '/media/meysam/B42683242682E6A8/OULU-NPU/'
        data_partion = 'train'
        train_dataset = OuluNPU(root,data_partion,cfg.train_batch_size,for_train=True,shape=shape)
        data_partion = 'devel'
        dev_dataset = OuluNPU(root,data_partion,cfg.devel_batch_size,for_train=False,shape=shape)
    elif cfg.dataset == 'replay':
        root = '/media/meysam/464C8BC94C8BB26B/Replay-Attack/' 
        # root = '/home/meysam/Desktop/Replay-Attack/'
        # root = '/content/replayattack/'
        
        data_partion = 'train'
        train_dataset = ReplayAttack(root,data_partion,cfg.train_batch_size,for_train=True,shape=shape)
        data_partion = 'devel'
        dev_dataset = ReplayAttack(root,data_partion,cfg.devel_batch_size,for_train=False,shape=shape)
    elif cfg.dataset == 'rose':
        root = '/media/meysam/464C8BC94C8BB26B/ROSE-YOUTU/'
        data_partion = 'train'
        train_dataset = RoseYoutu(root,data_partion,cfg.train_batch_size,for_train=True,shape=shape)
        data_partion = 'devel'
        dev_dataset = RoseYoutu(root,data_partion,cfg.devel_batch_size,for_train=False,shape=shape)
    elif cfg.dataset == 'siw':
        root = '/media/meysam/901292F51292E010/SiW/SiW_release/'
        data_partion = 'train'
        train_dataset = SiW(root,data_partion,cfg.train_batch_size,for_train=True,shape=shape)
        data_partion = 'devel'
        dev_dataset = SiW(root,data_partion,cfg.devel_batch_size,for_train=False,shape=shape)
    else:
        print("Error: unsuported datset!!")
    
    # tbs = TrainBatchSampler(train_dataset,cfg.train_batch_size)
    tbs = TrainBatchSampler4(train_dataset,cfg.train_batch_size)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size= 1, 
                               shuffle= False, sampler = None, batch_sampler = tbs,
                               num_workers= 0, collate_fn= None, pin_memory = False, 
                               drop_last = False, timeout= 0, worker_init_fn = None,
                               multiprocessing_context=None, generator=None, prefetch_factor = 2,
                               persistent_workers = False)
    
    dev_loader = data.DataLoader(dataset=dev_dataset, batch_size= cfg.devel_batch_size, 
                               shuffle= False, sampler = None, batch_sampler = None,
                               num_workers= 0, collate_fn= None, pin_memory = False, 
                               drop_last = False, timeout= 0, worker_init_fn = None,
                               multiprocessing_context=None, generator=None, prefetch_factor = 2,
                               persistent_workers = False)

    return train_loader,dev_loader

# 

def get_optimizer(net,cfg):
    if cfg.optimizer == 'adam':
        # optimizer = torch.optim.Adam(net.parameters(),lr=cfg.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        # optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=True)
        optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, dampening=0, weight_decay=0.001, nesterov=True)

    return optimizer

def get_criterion(cfg,net):
    if cfg.criterion == 'BCEWithLogits':
        criterion = BCEWithLogits()
    elif cfg.criterion == 'ArcB':
        criterion = ArcB(net,m=0.75,s=0.75)
    elif cfg.criterion == 'IdBce':
        criterion = IdBce(alpha=1,M=2)
    elif cfg.criterion == 'arcbid':
        # alpha,net,M = 0.5,m=0.5)
        criterion = ArcbId(alpha=2,beta=2,net=net,M=2,m=0.75)

    return criterion

def main():
    
    cfg = proc_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader,dev_loader = get_dataset(cfg)

    #net = get_net(cfg) 
    net = Model(cfg)

    net.to(device)

    # Our loss function
    criterion = get_criterion(cfg,net)
    # Our optimizer
    

    optimizer = get_optimizer(net,cfg)
    print('lr: '+str(cfg.lr))
    num_epochs = cfg.num_epochs
    path = cfg.dataset + '_' + cfg.backbone + '_' + cfg.criterion + '_' + cfg.optimizer+'_lbp_'+str(cfg.use_lbp)
    os.mkdir('outputs/'+path)
    os.mkdir('outputs/'+path+'/eer_figs')
    os.mkdir('outputs/'+path+'/checkpoints')

    log_file = open('outputs/'+path+'/logs.txt','w+')
    log_file.writelines('device: '+str(device)+'\n')
    log_file.writelines('lr: '+str(cfg.lr)+'\n')
    log_file.writelines('dataset: '+str(cfg.dataset)+'\n') # 
    log_file.writelines('backbone: '+str(cfg.backbone)+'\n')
    log_file.writelines('optimizer: '+str(cfg.optimizer)+'\n')
    log_file.writelines('criterion: '+str(cfg.criterion)+'\n')
    log_file.writelines('num_epochs: '+str(cfg.num_epochs)+'\n')
    log_file.writelines('use_lbp: '+str(cfg.use_lbp)+'\n')
    log_file.writelines('lbp_ch: '+str(cfg.lbp_ch)+'\n')
    log_file.writelines('train_batch_size: '+str(cfg.train_batch_size)+'\n')
    log_file.writelines('devel_batch_size: '+str(cfg.devel_batch_size)+'\n')
    log_file.writelines('emb_size: '+str(cfg.emb_size)+'\n')
    log_file.writelines('input_size: '+str(cfg.input_size)+'\n')
    log_file.writelines('----------------------------------------------------\n')
    for epoch in range(num_epochs):
        start = timeit.default_timer()
        train(net,criterion,optimizer,train_loader,device,epoch,path)
        train_loader.dataset.clear_cache()
        dev(net,criterion,dev_loader,device,epoch,path)
        stop = timeit.default_timer()
        print ('Epoch %d/%d, Tr Loss: %.10f, Dev Loss: %.10f,time: %.3f'
        %(epoch+1, num_epochs, train_loss[-1],dev_loss[-1], stop-start) )
        log_file.writelines('Epoch %d/%d, Tr Loss: %.10f, Dev Loss: %.10f,time: %.3f \n'
        %(epoch+1, num_epochs, train_loss[-1],dev_loss[-1], stop-start))
        torch.save(net.state_dict(), 'outputs/'+path+'/checkpoints/ep_'+str(epoch)+'.pt')
       

    plt.plot(list(range(1,num_epochs+1)),train_loss, label='train loss')
    plt.plot(list(range(1,num_epochs+1)),dev_loss, label='dev loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss per epochs')
    plt.grid()
    plt.savefig('outputs/'+path+'/loss.png')
    
    np.savetxt('outputs/'+path+'/train_loss.csv', np.array(train_loss), delimiter=',', fmt='%f')
    np.savetxt('outputs/'+path+'/dev_loss.csv', np.array(dev_loss), delimiter=',', fmt='%f')
    log_file.close()


if __name__ == "__main__":
    train_loss = []
    dev_loss = []
    main()
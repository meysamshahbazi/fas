import argparse
import torch.nn as nn
import torch
from protocol.protocol import calcEER
from dataset.sampler import TrainBatchSampler
from dataset.replayattack import ReplayAttack
from dataset.casiafasd import CasiaFASD
from dataset.msumfsd import MsuFsd
from dataset.oulunpu import OuluNPU
from dataset.roseyoutu import RoseYoutu
from dataset.siw import SiW
from model.alexnet import AlexNet,AlexNetLite
from loss.loss import BCEWithLogits,ArcB
from torch.utils import data
import matplotlib.pyplot as plt
import os
import timeit
import numpy as np




def train(net,criterion,optimizer,train_loader,device,epoch,path):
    global train_loss
    iter_loss = 0.0
    iterations = 0
    
    net.train()                   # Put the network into training mode
    optimizer.zero_grad()     # Clear off the gradients from any past operation
    for i, (items, classes) in enumerate(train_loader):
        # If we have GPU, shift the data to GPU

        items = items.to(device)
        classes = classes.to(device)

        optimizer.zero_grad()     # Clear off the gradients from any past operation
        
        #print(items.shape)
        outputs,emb = net(items)      # Do the forward pass
        #print("emb shape: "+str(emb.shape))
        #print(outputs.shape)
        #print(classes.shape)
        loss = criterion(outputs, classes,emb) # Calculate the loss
        #print(loss)
        iter_loss += loss.item()# Accumulate the loss
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

    for i, (items, classes) in enumerate(dev_loader):
        torch.cuda.empty_cache()
        # Convert torch tensor to Variable
        items = items.to(device)
        classes = classes.to(device)
        
        outputs,emb = net(items)      # Do the forward pass
        loss += criterion(outputs, classes,emb).item() # Calculate the loss
        outputs = sigmoid(outputs) # use sigmoid for infering
        # Record the all labels of dataset and prediction of model for later use!
        lbl += classes.cpu().flatten().tolist()    
        pred += outputs.detach().cpu().flatten().tolist()

        
        iterations += 1
    # Record the validation loss
    dev_loss.append(loss/iterations)
    FAR,FRR,threshold = calcEER(lbl,pred,epoch,path,plot=True,precision=0.01)



def proc_args():
    parser = argparse.ArgumentParser(description='base argument of train and dev loop')
    parser.add_argument(
    '--dataset',
    type = str,
    required=True,
    help='the dataset for train and dev which can be following:\n \
    casia\n \
    celeb\n \
    msu\n \
    oulu\n \
    replay\n \
    rose\n \
    siw\n '
    )
    parser.add_argument(
    '--model',
    type = str,
    required=True,
    help='the model for train and dev which can be following:\n \
    alexnet\n \
    lbpnet\n '
    )
    parser.add_argument(
    '--optimizer',
    type = str,
    default='adam',
    help='the optimizer for training which can be following:\n \
    adam\n \
    sgd\n '
    ) 
    parser.add_argument(
    '--criterion',
    type = str,
    default='BCEWithLogits',
    help='the criterion for claculation loss which can be following:\n \
    BCEWithLogits\n \
    ArcB\n \
    '
    )
    parser.add_argument(
    '--num_epochs',
    default=10,
    type = int,
    help='nubmers of epoch for runing train and dev (default: 10)'
    )
    parser.add_argument(
    '--lr',
    default=0.00001,
    type = float,
    help='learning rate of optimizer'
    )
    parser.add_argument(
    '--train_batch_size',
    default=8,
    type = int,
    help='batch size for train (default: 8)'
    )

    parser.add_argument(
    '--devel_batch_size',
    default=8,
    type = int,
    help='batch size for development (default: 8)'
    )

    
    parser.add_argument(
    '--path',
    default='./',
    type = str,
    help='path (default: 2)'
    )
    namespace = parser.parse_args()

    return namespace

def get_dataset(namespace):

    if namespace.dataset == 'casia':
        root = '/media/meysam/464C8BC94C8BB26B/Casia-FASD/'
        data_partion = 'train'
        train_dataset = CasiaFASD(root,data_partion,namespace.train_batch_size,for_train=True)
        data_partion = 'devel'
        dev_dataset = CasiaFASD(root,data_partion,namespace.devel_batch_size,for_train=False)
    elif namespace.dataset == 'msu':
        root = '/media/meysam/464C8BC94C8BB26B/MSU-MFSD/'
        data_partion = 'train'
        train_dataset = MsuFsd(root,data_partion,namespace.train_batch_size,for_train=True)
        data_partion = 'devel'
        dev_dataset = MsuFsd(root,data_partion,namespace.devel_batch_size,for_train=False)
    elif namespace.dataset == 'oulu':
        root = '/media/meysam/B42683242682E6A8/OULU-NPU/'
        data_partion = 'train'
        train_dataset = OuluNPU(root,data_partion,namespace.train_batch_size,for_train=True)
        data_partion = 'devel'
        dev_dataset = OuluNPU(root,data_partion,namespace.devel_batch_size,for_train=False)
    elif namespace.dataset == 'replay':
        root = '/media/meysam/464C8BC94C8BB26B/Replay-Attack/' 
        #root = '/home/meysam/Desktop/Replay-Attack/'
        data_partion = 'train'
        train_dataset = ReplayAttack(root,data_partion,namespace.train_batch_size,for_train=True)
        data_partion = 'devel'
        dev_dataset = ReplayAttack(root,data_partion,namespace.devel_batch_size,for_train=False)
    elif namespace.dataset == 'rose':
        root = '/media/meysam/464C8BC94C8BB26B/ROSE-YOUTU/'
        data_partion = 'train'
        train_dataset = RoseYoutu(root,data_partion,namespace.train_batch_size,for_train=True)
        data_partion = 'devel'
        dev_dataset = RoseYoutu(root,data_partion,namespace.devel_batch_size,for_train=False)
    elif namespace.dataset == 'siw':
        root = '/media/meysam/464C8BC94C8BB26B/ROSE-YOUTU/'
        data_partion = 'train'
        train_dataset = SiW(root,data_partion,namespace.train_batch_size,for_train=True)
        data_partion = 'devel'
        dev_dataset = SiW(root,data_partion,namespace.devel_batch_size,for_train=False)
    else:
        print("Error: unsuported datset!!")
    
    tbs = TrainBatchSampler(train_dataset,namespace.train_batch_size)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size= 1, 
                               shuffle= False, sampler = None, batch_sampler = tbs,
                               num_workers= 0, collate_fn= None, pin_memory = False, 
                               drop_last = False, timeout= 0, worker_init_fn = None,
                               multiprocessing_context=None, generator=None, prefetch_factor = 2,
                               persistent_workers = False)
    
    dev_loader = data.DataLoader(dataset=dev_dataset, batch_size= namespace.devel_batch_size, 
                               shuffle= False, sampler = None, batch_sampler = None,
                               num_workers= 0, collate_fn= None, pin_memory = False, 
                               drop_last = False, timeout= 0, worker_init_fn = None,
                               multiprocessing_context=None, generator=None, prefetch_factor = 2,
                               persistent_workers = False)

    return train_loader,dev_loader


def get_net(namespace):
    if  namespace.model == 'alexnet':
        net = AlexNetLite()
    #TODO: complete this
    return net
def get_optimizer(net,namespace):
    if namespace.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(),lr=namespace.lr)

    return optimizer

def get_criterion(namespace,net):
    if namespace.criterion == 'BCEWithLogits':
        criterion = BCEWithLogits()
    elif namespace.criterion == 'ArcB':
        criterion = ArcB(net,m=0.35)

    return criterion

def main():
    namespace = proc_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader,dev_loader = get_dataset(namespace)

    net = get_net(namespace) 
    net.to(device)

    # Our loss function
    criterion = get_criterion(namespace,net)
    # Our optimizer
    

    optimizer = get_optimizer(net,namespace)

    num_epochs = namespace.num_epochs
    path = namespace.dataset + '_' + namespace.model + '_' + namespace.criterion + '_' + namespace.optimizer
    os.mkdir('outputs/'+path)
    os.mkdir('outputs/'+path+'/eer_figs')
    os.mkdir('outputs/'+path+'/checkpoints')

    log_file = open('outputs/'+path+'/logs.txt','w+')
    log_file.writelines('device: '+str(device)+'\n')
    log_file.writelines('lr: '+str(namespace.lr)+'\n')
    log_file.writelines('dataset: '+str(namespace.dataset)+'\n') # 
    log_file.writelines('model: '+str(namespace.model)+'\n')
    log_file.writelines('optimizer: '+str(namespace.optimizer)+'\n')
    log_file.writelines('criterion: '+str(namespace.criterion)+'\n')
    log_file.writelines('num_epochs: '+str(namespace.num_epochs)+'\n')
    log_file.writelines('train_batch_size: '+str(namespace.train_batch_size)+'\n')
    log_file.writelines('devel_batch_size: '+str(namespace.devel_batch_size)+'\n')
    log_file.writelines('criterion: '+str(namespace.criterion)+'\n')
    log_file.writelines('criterion: '+str(namespace.criterion)+'\n')
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
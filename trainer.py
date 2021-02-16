import argparse
import torch.nn as nn
import torch
from protocol.protocol import calcEER
from dataset.sampler import TrainBatchSampler
from dataset.replayattack import ReplayAttack
from torch.utils import data
import matplotlib.pyplot as plt




def train(net,criterion,optimizer,train_loader,device):
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
        

        outputs = net(items)      # Do the forward pass

        loss = criterion(outputs, classes) # Calculate the loss
        iter_loss += loss.item()# Accumulate the loss
        loss.backward()           # Calculate the gradients with help of back propagation


        optimizer.step()          # Ask the optimizer to adjust the parameters based on the gradients

        iterations += 1
    # Record the training loss
    train_loss.append(iter_loss/iterations)



def dev(net,criterion,dev_loader,device):
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
        
        outputs = net(items)      # Do the forward pass
        loss += criterion(outputs, classes).item() # Calculate the loss
        outputs = sigmoid(outputs) # use sigmoid for infering
        # Record the all labels of dataset and prediction of model for later use!
        lbl += classes.cpu().flatten().tolist()    
        pred += outputs.detach().cpu().flatten().tolist()

        
        iterations += 1
    # Record the validation loss
    dev_loss.append(loss/iterations)
    FAR,FRR,threshold = calcEER(lbl,pred,plot=True,precision=0.01)



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
    '--num_epochs',
    default=10,
    type = int,
    help='nubmers of epoch for runing train and dev (default: 210)'
    )
    parser.add_argument(
    '--path',
    default='./',
    type = str,
    help='path (default: 2)'
    )
    my_namespace = parser.parse_args()
    print(type(my_namespace.my_optional))
    print(type(my_namespace.path)) 
    print(my_namespace.path[::-1])
    for i in range(my_namespace.my_optional):
        print(i**2)

    return my_namespace


def main():
    my_namespace = proc_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    all_datasets = {}
    
    data_partion = 'train'
    train_dataset = ReplayAttack(root,data_partion,batch_size,for_train=True)
    tbs = TrainBatchSampler(train_dataset,batch_size)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size= 1, 
                               shuffle= False, sampler = None, batch_sampler = tbs,
                               num_workers= 0, collate_fn= None, pin_memory = False, 
                               drop_last = False, timeout= 0, worker_init_fn = None,
                               multiprocessing_context=None, generator=None, prefetch_factor = 2,
                               persistent_workers = False)
    data_partion = 'dev'
    dev_dataset = ReplayAttack(root,data_partion,batch_size,for_train=False)
    dev_loader = data.DataLoader(dataset=dev_dataset, batch_size= 10, 
                               shuffle= False, sampler = None, batch_sampler = None,
                               num_workers= 0, collate_fn= None, pin_memory = False, 
                               drop_last = False, timeout= 0, worker_init_fn = None,
                               multiprocessing_context=None, generator=None, prefetch_factor = 2,
                               persistent_workers = False)

    net = AlexNet() 
    net.to(device)

    # Our loss function
    criterion = nn.BCEWithLogitsLoss()

    # Our optimizer
    lr = 0.00001 

    optimizer = torch.optim.Adam(net.parameters(),lr=lr)


    for epoch in range(num_epochs):
        train(net,criterion,optimizer,train_loader,device)
        train_dataset.clear_cache()
        dev(net,criterion,dev_loader,device)
        #TODO save results

    plt.plot(list(range(1,num_epochs+1)),train_loss, label='train loss')
    plt.plot(list(range(1,num_epochs+1)),dev_loss, label='dev loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss per epochs')
    plt.grid()
    plt.show()
    
    print("-------------------")
    print(my_namespace.dataset)

if __name__ == "__main__":
    train_loss = []
    dev_loss = []
    

    main()
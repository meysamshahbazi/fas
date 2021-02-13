import torch.nn as nn
import torch
from protocol.protocol import calcEER



def train(net,criterion,optimizer,train_loader,device):
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



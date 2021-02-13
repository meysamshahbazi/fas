import matplotlib.pyplot as plt
import numpy as np


def calcEER(lbl,pred,plot=True,precision=0.01):
  '''
  lbl: list of all labels in whole dataset
  pred: list of all prediction of model
  plot: flag for show resuts
  precision: precision of threshold in calculation 
  '''
  lbl = np.array(lbl)
  pred = np.array(pred)
  threshold = np.arange(0,1+precision,precision)
  FAR = np.zeros_like(threshold)
  FRR = np.zeros_like(threshold)
  for k,thr in enumerate(threshold):
    pre = np.where(pred>thr,1,0)
    FAR[k] = np.logical_and(pre==1,lbl ==0).sum()*100/(lbl==0).sum()
    FRR[k] = np.logical_and(pre==0,lbl ==1).sum()*100/(lbl==1).sum()


  index_EER = np.argmin(np.abs(FAR-FRR))
  if plot:#TODO: save results
    plt.plot(threshold,FAR,label='FAR')
    plt.plot(threshold,FRR,label='FRR')
    plt.vlines(threshold[index_EER],0,100,label='threshold: '+str(threshold[index_EER]),color='r',linestyles='dashed')
    plt.hlines((FAR[index_EER]+FRR[index_EER])/2,threshold.min(),threshold.max(),
              label='EER: '+str((FAR[index_EER]+FRR[index_EER])/2),color='g',linestyles='dashed')


    plt.legend()
    plt.xlabel('threshold')
    plt.ylabel('acc')
    plt.title(' acc per threshold')
    plt.grid()
    plt.show()

    return FAR,FRR,threshold[index_EER]


    
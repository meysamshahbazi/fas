import matplotlib.pyplot as plt
import numpy as np


def calcEER(lbl,pred,epoch,path,plot=True,precision=0.01):
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


  #index_EER = np.argmin(np.abs(FAR-FRR))
  indecis_of_min = np.where(np.abs(FAR-FRR)==np.min(np.abs(FAR-FRR)))[0]
  index_EER = indecis_of_min[len(indecis_of_min)//2] # in this way we got the medina index of EER

  if plot:
    plt.plot(threshold,FAR,label='FAR')
    plt.plot(threshold,FRR,label='FRR')
    plt.vlines(threshold[index_EER],0,100,label='threshold: '+str(threshold[index_EER]),color='r',linestyles='dashed')
    plt.hlines((FAR[index_EER]+FRR[index_EER])/2,threshold.min(),threshold.max(),
              label='EER: '+str((FAR[index_EER]+FRR[index_EER])/2),color='g',linestyles='dashed')


    plt.legend()
    plt.xlabel('threshold')
    plt.ylabel('err')
    plt.title(' err per threshold')
    plt.grid()
    plt.savefig('outputs/'+path+'/eer_figs/ep_'+str(epoch)+'.png')
    plt.close()
    np.savetxt('outputs/'+path+'/eer_figs/ep_'+str(epoch)+'_FAR.csv', np.array(FAR), delimiter=',', fmt='%f')
    np.savetxt('outputs/'+path+'/eer_figs/ep_'+str(epoch)+'_FRR.csv', np.array(FRR), delimiter=',', fmt='%f')
    # plt.show()

    return FAR,FRR,threshold[index_EER]


def calcHTER(lbl,pred,threshold,path):
  lbl = np.array(lbl)
  pred = np.array(pred)
  pre = np.where(pred>threshold,1,0)
  print(lbl.shape)
  print(pre.shape)
  FAR = np.logical_and(pre==1,lbl ==0).sum()*100/(lbl==0).sum()
  FRR = np.logical_and(pre==0,lbl ==1).sum()*100/(lbl==1).sum()
  HTER = (FAR+FRR)/2
  print("HTER: ",HTER)
  log_file = open('outputs/'+path+'/HTER.txt','w+')
  log_file.writelines("HTER: "+str(HTER)+"\n")
  log_file.close()
  return HTER

  

from dataset.sampler import TrainBatchSampler,TrainBatchSampler2,TrainBatchSampler3,TrainBatchSampler4
from dataset.replayattack import ReplayAttack
from dataset.casiafasd import CasiaFASD
from dataset.msumfsd import MsuFsd
from dataset.oulunpu import OuluNPU
from dataset.roseyoutu import RoseYoutu
from dataset.siw import SiW
from torch.utils import data



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

    
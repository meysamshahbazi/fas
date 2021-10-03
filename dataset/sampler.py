import torch
from torch.utils import data
import random


class TrainBatchSampler(data.sampler.Sampler):
    '''
    this is my batch sampler for train 
    in this way we select batch size video and them select randomly min_nb_frame
    for each video and then select another videso batch
    with this method in each batch we reuse preloaded video frames
    this is a little far from complete randomness! but the speed will 
    increase in order of magnitude!

    '''
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.min_frames = self.dataset.datadict['0']['nb_frame']
        for k in self.dataset.datadict.keys():
            if self.dataset.datadict[k]['nb_frame'] < self.min_frames:
                self.min_frames = self.dataset.datadict[k]['nb_frame']
    def __iter__(self):
        all_vids = list(range(len(self.dataset.datadict)))
        random.shuffle(all_vids)
        vids_for_batch = torch.split(torch.tensor(all_vids),self.batch_size)
        vids_for_batch = [batch.tolist() for batch in vids_for_batch]
        
        vids_iter = [[ (v,iter(data.SubsetRandomSampler(range(self.dataset.datadict[str(v)]['nb_frame']))) ) for v in batch] for batch in vids_for_batch]
        
        for i in range(len(self.dataset.datadict)*self.min_frames// self.batch_size):
            yield [(v,next(g)) for v,g in vids_iter[i//self.min_frames] ] 
        
    def __len__(self):
        return len(self.dataset.datadict)*self.min_frames// self.batch_size


class TrainBatchSampler2(data.sampler.Sampler):
    '''
    this is my batch sampler for train 
    in this way we select batch size video and them select randomly min_nb_frame
    for each video and then select another videso batch
    with this method in each batch we reuse preloaded video frames
    this is a little far from complete randomness! but the speed will 
    increase in order of magnitude!

    '''
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.max_frames = self.dataset.datadict['0']['nb_frame']
        for k in self.dataset.datadict.keys():
            if self.dataset.datadict[k]['nb_frame'] > self.max_frames:
                self.max_frames = self.dataset.datadict[k]['nb_frame']
        
        # self.max_frames = self.max_frames//2 # just for siw 
        print("max frame:",self.max_frames)

    def __iter__(self):
        all_vids = list(range(len(self.dataset.datadict)))
        random.shuffle(all_vids)
        vids_for_batch = torch.split(torch.tensor(all_vids),self.batch_size)
        vids_for_batch = [batch.tolist() for batch in vids_for_batch]
        
        vids_iter = [[ (v,iter(data.SubsetRandomSampler(range(self.max_frames))) ) for v in batch] for batch in vids_for_batch]
        
        # for i in range(len(self.dataset.datadict)*self.max_frames// self.batch_size):
        for i in range((len(self.dataset.datadict)// self.batch_size +1)*self.max_frames):

            yield [(v,next(g)) for v,g in vids_iter[i//self.max_frames ] ] 
        
    def __len__(self):
        return (len(self.dataset.datadict)// self.batch_size +1)*self.max_frames
        # return len(self.dataset.datadict)*self.max_frames// self.batch_size


# random.sample(range(10, 30), 5)

class TrainBatchSampler3(data.sampler.Sampler):
    '''
    this is my batch sampler for train 
    in this way we select batch size video and them select randomly min_nb_frame
    for each video and then select another videso batch
    with this method in each batch we reuse preloaded video frames
    this is a little far from complete randomness! but the speed will 
    increase in order of magnitude!

    '''
    def __init__(self, dataset, batch_size,frame_per_vid=300):
        self.batch_size = batch_size
        self.dataset = dataset
        self.max_frames = self.dataset.datadict['0']['nb_frame']
        for k in self.dataset.datadict.keys():
            if self.dataset.datadict[k]['nb_frame'] > self.max_frames:
                self.max_frames = self.dataset.datadict[k]['nb_frame']
        
        # self.max_frames = self.max_frames//2 # just for siw 
        # self.max_frames = 30
        self.frame_per_vid = frame_per_vid
        print("max frame:",self.max_frames)

    def __iter__(self):
        all_vids = list(range(len(self.dataset.datadict)))
        random.shuffle(all_vids)
        vids_for_batch = torch.split(torch.tensor(all_vids),self.batch_size)
        vids_for_batch = [batch.tolist() for batch in vids_for_batch]
        
        vids_iter = [[ (v,iter(data.SubsetRandomSampler(range(self.max_frames))) ) for v in batch] for batch in vids_for_batch]
        # vids_iter = [[ (v,iter(data.SubsetRandomSampler(range(self.dataset.datadict[str(v)]['nb_frame']))) ) for v in batch] for batch in vids_for_batch]
        
        # for i in range(len(self.dataset.datadict)*self.max_frames// self.batch_size):
        for i in range((len(self.dataset.datadict)// self.batch_size +1)*self.frame_per_vid):

            yield [(v,next(g)) for v,g in vids_iter[i//self.frame_per_vid ] ] 
        
    def __len__(self):
        return (len(self.dataset.datadict)// self.batch_size +1)*self.frame_per_vid




class TrainBatchSampler4(data.sampler.Sampler):
    '''
    this is my batch sampler for train 
    in this way we select batch size video and them select randomly min_nb_frame
    for each video and then select another videso batch
    with this method in each batch we reuse preloaded video frames
    this is a little far from complete randomness! but the speed will 
    increase in order of magnitude!

    '''
    def __init__(self, dataset, batch_size,frame_per_vid=300):
        self.batch_size = batch_size
        self.dataset = dataset
        self.max_frames = self.dataset.datadict['0']['nb_frame']
        for k in self.dataset.datadict.keys():
            if self.dataset.datadict[k]['nb_frame'] > self.max_frames:
                self.max_frames = self.dataset.datadict[k]['nb_frame']
        
        # self.max_frames = self.max_frames//2 # just for siw 
        # self.max_frames = 300
        self.frame_per_vid = self.max_frames
        # self.frame_per_vid = frame_per_vid
        print("max frame:",self.max_frames)
        
        all_vids = list(range(len(self.dataset.datadict)))
        pos_vids = list(filter(lambda v: int(self.dataset.datadict[str(v)]['real_or_spoof']) == 1, all_vids))
        neg_vids = list(filter(lambda v: int(self.dataset.datadict[str(v)]['real_or_spoof']) == 0, all_vids))
        print(len(list(pos_vids)))
        print(len(list(neg_vids)))
        print(len(all_vids))
        
        if len(pos_vids) > len(neg_vids):
            self.max_vids = pos_vids
            self.min_vids = neg_vids
            
        else:
            self.max_vids = neg_vids
            self.min_vids = pos_vids
            
        
    def __iter__(self):
        
        
        all_vids = []
        random.shuffle(self.max_vids)
        random.shuffle(self.min_vids)
        for i,m in enumerate(self.max_vids): # for siw default is max!!
            all_vids.append(m)
            all_vids.append(self.min_vids[i%len(self.min_vids)])
            
        vids_for_batch = torch.split(torch.tensor(all_vids),self.batch_size)
        vids_for_batch = [batch.tolist() for batch in vids_for_batch]
        
        # vids_iter = [[ (v,iter(data.SubsetRandomSampler(range(max(self.dataset.datadict[str(v)]['nb_frame'],self.max_frames))) ) for v in batch] for batch in vids_for_batch]
        # vids_iter = [[ (v,iter(data.SubsetRandomSampler(range( max(self.dataset.datadict[str(v)]['nb_frame'],self.max_frames) ))) ) for v in batch] for batch in vids_for_batch]
        vids_iter = [[ (v,iter(data.SubsetRandomSampler(range(self.frame_per_vid))) ) for v in batch] for batch in vids_for_batch]
        for i in range(math.ceil(2*len(self.max_vids)/ self.batch_size)*self.frame_per_vid):
            yield [(v,next(g)) for v,g in vids_iter[i//self.frame_per_vid ] ] # for siw default is max!!
        
    def __len__(self):
        return math.ceil(2*len(self.max_vids)/ self.batch_size)*self.frame_per_vid # for siw default is max!!




class TrainBatchSampler5(data.sampler.Sampler):
    '''
    this is my batch sampler for train 
    in this way we select batch size video and them select randomly min_nb_frame
    for each video and then select another videso batch
    with this method in each batch we reuse preloaded video frames
    this is a little far from complete randomness! but the speed will 
    increase in order of magnitude!

    '''
    def __init__(self, dataset, batch_size,frame_per_vid=300):
        self.batch_size = batch_size
        self.dataset = dataset
        self.max_frames = self.dataset.datadict['0']['nb_frame']
        for k in self.dataset.datadict.keys():
            if self.dataset.datadict[k]['nb_frame'] > self.max_frames:
                self.max_frames = self.dataset.datadict[k]['nb_frame']
        
        # self.max_frames = self.max_frames//2 # just for siw 
        # self.max_frames = 300
        # self.frame_per_vid = self.max_frames
        self.frame_per_vid = frame_per_vid
        print("max frame:",self.max_frames)
        
        all_vids = list(range(len(self.dataset.datadict)))
        pos_vids = list(filter(lambda v: int(self.dataset.datadict[str(v)]['real_or_spoof']) == 1, all_vids))
        neg_vids = list(filter(lambda v: int(self.dataset.datadict[str(v)]['real_or_spoof']) == 0, all_vids))
        print(len(list(pos_vids)))
        print(len(list(neg_vids)))
        print(len(all_vids))
        
        if len(pos_vids) > len(neg_vids):
            self.max_vids = pos_vids
            self.min_vids = neg_vids
            
        else:
            self.max_vids = neg_vids
            self.min_vids = pos_vids
            
        
    def __iter__(self):
        
        
        all_vids = []
        random.shuffle(self.max_vids)
        random.shuffle(self.min_vids)
        for i,m in enumerate(self.min_vids): # for siw default is max!!
            all_vids.append(m)
            all_vids.append(self.min_vids[i%len(self.min_vids)])
            
        vids_for_batch = torch.split(torch.tensor(all_vids),self.batch_size)
        vids_for_batch = [batch.tolist() for batch in vids_for_batch]
        
        # vids_iter = [[ (v,iter(data.SubsetRandomSampler(range(max(self.dataset.datadict[str(v)]['nb_frame'],self.max_frames))) ) for v in batch] for batch in vids_for_batch]
        # vids_iter = [[ (v,iter(data.SubsetRandomSampler(range( max(self.dataset.datadict[str(v)]['nb_frame'],self.max_frames) ))) ) for v in batch] for batch in vids_for_batch]
        vids_iter = [[ (v,iter(data.SubsetRandomSampler(range(self.max_frames))) ) for v in batch] for batch in vids_for_batch]
        for i in range(math.ceil(2*len(self.min_vids)/ self.batch_size)*self.frame_per_vid):
            yield [(v,next(g)) for v,g in vids_iter[i//self.frame_per_vid ] ] # for siw default is max!!
        
    def __len__(self):
        return math.ceil(2*len(self.min_vids)/ self.batch_size)*self.frame_per_vid # for siw default is max!!

        


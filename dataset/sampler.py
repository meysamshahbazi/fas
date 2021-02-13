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



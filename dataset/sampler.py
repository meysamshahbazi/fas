import torch
from torch.utils import data
import random
import math




class TrainBatchSampler(data.sampler.Sampler):
    '''
    this is my batch sampler for train 
    in this way we select batch size video and them select randomly min_nb_frame
    for each video and then select another videso batch
    with this method in each batch we reuse preloaded video frames
    this is a little far from complete randomness! but the speed will 
    increase in order of magnitude!

    '''
    def __init__(self, dataset, batch_size,frame_per_vid='min'):
        super(TrainBatchSampler, self).__init__(None)
        self.batch_size = batch_size
        self.dataset = dataset
        frame_nbs = [self.dataset.datadict[k]['nb_frame'] for k in self.dataset.datadict.keys() ]
        self.max_frames = max(frame_nbs)
        self.min_frames = min(frame_nbs)


        if frame_per_vid == 'max':
            self.frame_per_vid = self.max_frames
        elif frame_per_vid == 'min':
            self.frame_per_vid = self.min_frames
        else:
            self.frame_per_vid = frame_per_vid

        all_vids = list(range(len(self.dataset.datadict)))
        pos_vids = list(filter(lambda v: int(self.dataset.datadict[str(v)]['real_or_spoof']) == 1, all_vids))
        neg_vids = list(filter(lambda v: int(self.dataset.datadict[str(v)]['real_or_spoof']) == 0, all_vids))



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
        for i,m in enumerate(self.max_vids): 
            all_vids.append(m)
            all_vids.append(self.min_vids[i%len(self.min_vids)])
            
        vids_for_batch = torch.split(torch.tensor(all_vids),self.batch_size)
        vids_for_batch = [batch.tolist() for batch in vids_for_batch]
        
        vids_iter = [[ (v,iter(data.SubsetRandomSampler(range(self.max_frames))) ) for v in batch] for batch in vids_for_batch]

        for i in range(math.ceil(2*len(self.max_vids)/ self.batch_size)*self.frame_per_vid):
            yield [(v,next(g)) for v,g in vids_iter[i//self.frame_per_vid ] ] 
        
    def __len__(self):
        return math.ceil(2*len(self.max_vids)/ self.batch_size)*self.frame_per_vid 

    def __str__(self):
        return "RANDAOM SAMPLER | batch_size: "+str(self.batch_size)+" |max_frames: "+str( self.max_frames ) + " |min_frames: "+str(self.min_frames)+" |frame_per_vid: " + str(self.frame_per_vid) 
 



# imports
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchaudio
import os
import settings as s


class AudioDataset(Dataset):
    '''audio dataset class'''
    def __init__(self, low_quality_paths, high_quality_paths):
        self.low_quality_paths = low_quality_paths
        self.high_quality_paths = high_quality_paths

    def __len__(self):
        return len(self.low_quality_paths)

    def __getitem__(self, idx):
        low_quality, _ = torchaudio.load(self.low_quality_paths[idx])
        high_quality, _ = torchaudio.load(self.high_quality_paths[idx])
        
        #round to 3 decimals by multiplying 
        #by 1000 and discarding decimal

        low_quality = ((low_quality*1000).int()/1000).float()
        high_quality = ((high_quality*1000).int()/1000).float()

        return low_quality, high_quality

def load_audio_dataset():
    '''set up all audio datasets to return them'''
    # Set up the data paths 
    train_low_quality_paths = os.listdir(s.LOW_QUALITY_PATH) 
    train_high_quality_paths = os.listdir(s.HIGH_QUALITY_PATH) 
    val_low_quality_paths = os.listdir(s.VAL_LQ) 
    val_high_quality_paths = os.listdir(s.VAL_HQ) 
    test_low_quality_paths = os.listdir(s.TEST_DATA_LQ) 
    test_high_quality_paths = os.listdir(s.TEST_DATA_HQ) 

    for i in range(0, len(train_low_quality_paths)):
        train_low_quality_paths[i] = s.LOW_QUALITY_PATH + train_low_quality_paths[i]
        train_high_quality_paths[i] = s.HIGH_QUALITY_PATH + train_high_quality_paths[i]

    for i in range(0, len(val_low_quality_paths)):
        val_low_quality_paths[i] = s.VAL_LQ + val_low_quality_paths[i]
        val_high_quality_paths[i] = s.VAL_HQ + val_high_quality_paths[i]

    for i in range(0, len(test_low_quality_paths)):
        test_low_quality_paths[i] = s.TEST_DATA_LQ + test_low_quality_paths[i]
        test_high_quality_paths[i] = s.TEST_DATA_HQ + test_high_quality_paths[i]

    # Set up the data loaders
    # pair low and high quality data
    train_dataset = AudioDataset(train_low_quality_paths, train_high_quality_paths)
    val_dataset = AudioDataset(val_low_quality_paths, val_high_quality_paths)
    test_dataset = AudioDataset(test_low_quality_paths, test_high_quality_paths)

    train_dataloader = DataLoader(train_dataset, batch_size=s.BATCH_SIZE, shuffle=True, collate_fn=collate_function)
    val_dataloader = DataLoader(val_dataset, batch_size=s.BATCH_SIZE, shuffle=False, collate_fn= collate_function)
    test_dataloader = DataLoader(test_dataset, batch_size=s.BATCH_SIZE, shuffle=False, collate_fn= collate_function)
    
    return train_dataloader, val_dataloader, test_dataloader

def get_directory_paths(directory_path, start_percent, stop_percent):
    '''was originally used to get data 
    paths but they have been
    seperated into seperate folders'''
    all_paths = os.listdir(directory_path)
    numpaths = len(all_paths)
    # Calculate the index that represents the % point
    start_index =int(numpaths * start_percent)
    stop_index = int(numpaths * stop_percent)
    
    # Slice the list to get the first % of paths
    first_percent = all_paths[start_index:stop_index]
    
    for i in range(0,len(first_percent)):
        first_percent[i] = directory_path + first_percent[i]
        
    return first_percent

def collate_function(batch):
    '''Function to pad each item in a batch to be the same length.
    Takes the longest audio in a batch and pads every 
    other audio to be the same length'''
    # print(batch)
    #sort by size in descending order
    sorted_batch = sorted(batch, key = lambda x: x[1].size(), reverse=True)

    #seperate high and low quality for processing
    low_quality = [x[0] for x in sorted_batch]
    #[batch, input/target, length]
    high_quality = [x[1] for x in sorted_batch]
    
    # #check sizes before
    # for item in low_quality:
    #     print(item.size())
    
    #get the size of the biggest sequence
    max_size = low_quality[0].size(1)
    padded_low_quality = []
    padded_high_quality = []
    #manual padding
    for i in range(0, len(low_quality)):
        padding = max_size - low_quality[i].size(1)
        padded_item = torch.nn.functional.pad(low_quality[i], (0, padding), value=0.0)
        padded_low_quality.append(padded_item)
        
        # print(padded_item)
        
        hq_max_size = max_size*6
        padding = hq_max_size - high_quality[i].size(1)
        padded_item = torch.nn.functional.pad(high_quality[i], (0, padding), value=0.0)
        padded_high_quality.append(padded_item)
        
    # #check sizes after
    # for item in padded_low_quality:
    #     print(item.size())
    # print(type(padded_low_quality[0]))
    
    #stack into new dim

    stacked_lq = torch.stack(*[padded_low_quality], dim=0)
    stacked_hq = torch.stack(*[padded_high_quality], dim=0)
    
    # print(stacked_lq.size())
    # print(zip(low_quality, high_quality))
    # out = torch.transpose(padded_out, 0, 1)
    
    # print(out)
    stacked_lq = torch.transpose(stacked_lq, 2, 1)
    stacked_hq = torch.transpose(stacked_hq, 2, 1)
    # print(stacked_lq.size())
    return stacked_lq, stacked_hq
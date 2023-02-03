import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MovingMNistDataset:
    def __init__(self, data):
        self.data = data
        #(batch,seq,height,width)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        get_data = self.data[idx]
        #print(f"shape of get_data: {get_data.shape}")
        x_frames = self.data[idx][:10]
        y_frames = self.data[idx][10:]
        return x_frames, y_frames
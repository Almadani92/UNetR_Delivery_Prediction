# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
from PIL import Image

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks, savgol_filter

import wfdb


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

"""# Load Dataset"""

class Dataset(data.Dataset):

    def __init__(self, root='./', load_set='train', transform=None):
        self.root = root#os.path.expanduser(root)
        self.transform = transform
        self.load_set = load_set  # 'train','val','test'
        


        self.inputs_paths = np.load(os.path.join(root, 'inputs_paths_%s.npy'%self.load_set))
        self.labels_paths = np.load(os.path.join(root, 'labels_paths_%s.npy'%self.load_set))
        
        
        #if shuffle:
        #    random.shuffle(data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """
        # print(self.inputs_paths[0])
        inputs = np.load(self.inputs_paths[index], allow_pickle=True)
        labels = np.load(self.labels_paths[index], allow_pickle=True)
        
        # inputs = (inputs-np.mean(inputs))/np.var(inputs) 
        
        # inputs = inputs[:,4]
        # b = np.min(inputs)
        # inputs = (inputs - b) 
        # q = np.max(inputs)
        # inputs = inputs / q
        # inputs = np.expand_dims(inputs, axis=1)    
        
        inputs[:,0] =butter_bandpass_filter(inputs[:,0], 0.5, 90, 330, 5)
        # inputs[:,0] = (inputs[:,0]-np.mean(inputs[:,0]))/np.var(inputs[:,0]) 
        
        inputs[:,1] =butter_bandpass_filter(inputs[:,1], 0.5, 90, 330, 5)
        # inputs[:,1] = (inputs[:,1]-np.mean(inputs[:,1]))/np.var(inputs[:,1]) 
        
        inputs[:,2] =butter_bandpass_filter(inputs[:,2], 0.5, 90, 330, 5)
        # inputs[:,2] = (inputs[:,2]-np.mean(inputs[:,2]))/np.var(inputs[:,2]) 
        
        inputs[:,3] =butter_bandpass_filter(inputs[:,3], 0.5, 90, 330, 5)
        # inputs[:,3] = (inputs[:,3]-np.mean(inputs[:,3]))/np.var(inputs[:,3]) 
        
        inputs[:,4] =butter_bandpass_filter(inputs[:,4], 0.5, 90, 330, 5)
        # inputs[:,4] = (inputs[:,4]-np.mean(inputs[:,4]))/np.var(inputs[:,4]) 
        
        b = np.min(inputs[:,0])
        inputs[:,0] = (inputs[:,0] - b) 
        q = np.max(inputs[:,0])
        inputs[:,0] = inputs[:,0] / q
        
        
        b = np.min(inputs[:,1])
        inputs[:,1] = (inputs[:,1] - b) 
        q = np.max(inputs[:,1])
        inputs[:,1] = inputs[:,1] / q
        
        
        b = np.min(inputs[:,2])
        inputs[:,2] = (inputs[:,2] - b) 
        q = np.max(inputs[:,2])
        inputs[:,2] = inputs[:,2] / q
        
        
        b = np.min(inputs[:,3])
        inputs[:,3] = (inputs[:,3] - b) 
        q = np.max(inputs[:,3])
        inputs[:,3] = inputs[:,3] / q
        
        
        b = np.min(inputs[:,4])
        inputs[:,4] = (inputs[:,4] - b) 
        q = np.max(inputs[:,4])
        inputs[:,4] = inputs[:,4] / q

        
        # print("max and min of mixuter is ->>>", np.min(mixture),"--------------", np.max(mixture))
        # inputs_new = np.zeros([inputs.shape[0],2])
        # inputs_new[:,0:2] = inputs[:,0:2]
        # inputs_new[:,2] = inputs[:,4]

        return inputs, labels

    def __len__(self):
        return len(self.inputs_paths)

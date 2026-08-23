# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks, savgol_filter
import numpy as np
import os
from scipy.signal import welch
# import wfdb
import random

def generate_random_numbers(num_times, lower_bound, upper_bound):
    """
    Generates 'num_times' random numbers between 'lower_bound' and 'upper_bound'.
    """
    return [random.randint(lower_bound, upper_bound) for _ in range(num_times)]

frequency_range = (1, 100)
# thr=0.95
thr=0.9
# thr=0.85
# thr=0
# Function to calculate SNR-based quality index for a given ECG signal
def calculate_snr_quality(ecg_signal):
    snr_quality_per_lead = np.zeros([5])
    for i in range(5):
        lead_data = ecg_signal[:, i]
        # Calculate power spectral density (PSD) using Welch's method
        f, power_ecg = welch(lead_data, fs=333, nperseg=256)
        
        # Extract power within the specified frequency range
        power_range = np.sum(power_ecg[(frequency_range[0] <= f) & (f <= frequency_range[1])])
        
        # Calculate total power of the signal
        total_power = np.sum(power_ecg)
        
        # Calculate SNR-based quality index
        snr_quality = power_range / total_power
        snr_quality_per_lead[i] = snr_quality
    mean_snr_quality = np.mean(snr_quality_per_lead)
        
    return mean_snr_quality

random_numbers1 = [13,2,3,16,17]
# random_numbers1 = [4,5,6]
# random_numbers1 = [14,1,9]
# random_numbers1 = [10,11,12]
# random_numbers1 = [8,7,15]


random_numbers2 = range(1,12)
# random_numbers2 = range(12,24)
# random_numbers2 = range(24,36)
# random_numbers2 = range(36,48)
# random_numbers2 = range(48,58)






inputs_paths_val = []
inputs_paths_train = []

labels_paths_val = []
labels_paths_train = []



for i in range(1,67):   
    s0=0
    for kh in range(0,301):
        inputs_path = 'Dataset/signal/0/'+'signal_'+str(i)+'_'+str(kh)+'.npy'
        labels_path = 'Dataset/class_num/0/'+'class_num_'+str(i)+'_'+str(kh)+'.npy'
        inp_path = 'signal/0/'+'signal_'+str(i)+'_'+str(kh)+'.npy'
        inputs = np.load(inp_path)
        snr = calculate_snr_quality(inputs)  
        if i in random_numbers2:
            if snr>thr:
                s0=s0+1
                inputs_paths_val.append(inputs_path)
                labels_paths_val.append(labels_path)      
        else:
            # if snr>thr:
            inputs_paths_train.append(inputs_path)
            labels_paths_train.append(labels_path)
    if i in random_numbers2:
        print("The number of samples in subject ", i, "is -------------->", s0)
            
for i in range(1,16):              
    s1=0
    for kh in range(0,301):
        inputs_path = 'Dataset/signal/1/'+'signal_'+str(i)+'_'+str(kh)+'.npy'
        # subject_no_path = 'Dataset/signal_subject_no/1/'+'subject_'+str(i)+'_'+str(kh)+'.npy'
        labels_path = 'Dataset/class_num/1/'+'class_num_'+str(i)+'_'+str(kh)+'.npy'
        inp_path = 'signal/1/'+'signal_'+str(i)+'_'+str(kh)+'.npy'
        inputs = np.load(inp_path)
        snr = calculate_snr_quality(inputs) 
        if i in random_numbers1:
            if snr>thr:
                s1=s1+1
                inputs_paths_val.append(inputs_path)
                labels_paths_val.append(labels_path)      
        else:
            # if snr>thr:
            inputs_paths_train.append(inputs_path)
            labels_paths_train.append(labels_path)
    if i in random_numbers1:
        print("The number of samples in subject ", i, "is -------------->", s1)

inputs_paths_val = np.array(inputs_paths_val)
inputs_paths_train = np.array(inputs_paths_train)
labels_paths_val = np.array(labels_paths_val)
labels_paths_train = np.array(labels_paths_train)

np.save('inputs_paths_val.npy',inputs_paths_val)
np.save('inputs_paths_train.npy',inputs_paths_train)
np.save('labels_paths_val.npy',labels_paths_val)
np.save('labels_paths_train.npy',labels_paths_train)

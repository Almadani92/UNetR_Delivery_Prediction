# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
import scipy
import scipy.io
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks, savgol_filter
import numpy as np
import os
from scipy.signal import welch
# import mne
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
    

folder_path = 'processed_data1'  # Replace with the path to your folder

# Get a list of all files in the folder
file_list = os.listdir(folder_path)
# Filter the list to include only .mat files
mat_files = [file for file in file_list if file.endswith('.mat')]

i=0
for file in mat_files:
    file_path = os.path.join(folder_path, file)
    mat = scipy.io.loadmat(file_path)
    data = mat['ecgSignals']
    class_num = mat['delivery']
    # masqi = mat['masqi']
    # fsqi = mat['fsqi']
    # snr = mat['snr']
    # sqi = fsqi+masqi
    # if snr>thr:
    if class_num==0:
        i=i+1
        print(file,'--------------------------------',i)
        for kh in range(0,301):
            # data[1984*kh:1984*(kh+1),0] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),0], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),1] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),1], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),2] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),2], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),3] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),3], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),4] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),4], 1, 90, 330, 5)
            np.save('signal/0/signal_'+str(i)+'_'+str(kh), data[1984*kh:1984*(kh+1)])
            np.save('class_num/0/class_num_'+str(i)+'_'+str(kh), class_num)
            # np.save('signal_subject_no/0/subject_'+str(i)+'_'+str(kh), i)
            
            
folder_path = 'processed_data2'  # Replace with the path to your folder

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Filter the list to include only .mat files
mat_files = [file for file in file_list if file.endswith('.mat')]
for file in mat_files:
    file_path = os.path.join(folder_path, file)
    mat = scipy.io.loadmat(file_path)
    data = mat['ecgSignals']
    class_num = mat['delivery']
    # masqi = mat['masqi']
    # fsqi = mat['fsqi']
    # snr = mat['snr']
    # sqi = fsqi+masqi
    # if snr>thr:
    if class_num==0:
        i=i+1
        print(file,'--------------------------------',i)
        for kh in range(0,301):
            # data[1984*kh:1984*(kh+1),0] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),0], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),1] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),1], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),2] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),2], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),3] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),3], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),4] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),4], 1, 90, 330, 5)
            np.save('signal/0/signal_'+str(i)+'_'+str(kh), data[1984*kh:1984*(kh+1)])
            np.save('class_num/0/class_num_'+str(i)+'_'+str(kh), class_num)
            # np.save('signal_subject_no/0/subject_'+str(i)+'_'+str(kh), i)


folder_path = 'processed_data1'  # Replace with the path to your folder

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Filter the list to include only .mat files
mat_files = [file for file in file_list if file.endswith('.mat')]    
i=0
for file in mat_files:
    file_path = os.path.join(folder_path, file)
    mat = scipy.io.loadmat(file_path)
    data = mat['ecgSignals']
    class_num = mat['delivery'] 
    # masqi = mat['masqi']
    # fsqi = mat['fsqi']  
    # snr = mat['snr']
    # sqi = fsqi+masqi
    # if snr>thr:    
    if class_num==1:
        i=i+1
        print(file,'--------------------------------',i)
        for kh in range(0,301):
            # data[1984*kh:1984*(kh+1),0] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),0], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),1] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),1], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),2] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),2], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),3] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),3], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),4] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),4], 1, 90, 330, 5)
            np.save('signal/1/signal_'+str(i)+'_'+str(kh), data[1984*kh:1984*(kh+1)])
            np.save('class_num/1/class_num_'+str(i)+'_'+str(kh), class_num)
            # np.save('signal_subject_no/1/subject_'+str(i)+'_'+str(kh), i)
        
folder_path = 'processed_data2'  # Replace with the path to your folder

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Filter the list to include only .mat files
mat_files = [file for file in file_list if file.endswith('.mat')]
for file in mat_files:
    file_path = os.path.join(folder_path, file)
    mat = scipy.io.loadmat(file_path)
    data = mat['ecgSignals']
    class_num = mat['delivery']  
    # masqi = mat['masqi']
    # fsqi = mat['fsqi']    
    # snr = mat['snr']
    # sqi = fsqi+masqi
    # if snr>thr:
    if class_num==1:
        i=i+1
        print(file,'--------------------------------',i)
        for kh in range(0,301):
            # data[1984*kh:1984*(kh+1),0] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),0], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),1] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),1], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),2] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),2], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),3] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),3], 1, 90, 330, 5)
            # data[1984*kh:1984*(kh+1),4] =butter_bandpass_filter(data[1984*kh:1984*(kh+1),4], 1, 90, 330, 5)
            np.save('signal/1/signal_'+str(i)+'_'+str(kh), data[1984*kh:1984*(kh+1)])
            np.save('class_num/1/class_num_'+str(i)+'_'+str(kh), class_num)
            # np.save('signal_subject_no/1/subject_'+str(i)+'_'+str(kh), i)
                
                
                
                

    
    

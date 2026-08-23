# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
import scipy
import scipy.io
from scipy import signal
import numpy as np
import os
import matplotlib.pyplot as plt



file_path = 'processed_data2/7.mat'  # Replace with the path to your folder
mat = scipy.io.loadmat(file_path)
data = mat['ecgSignals']
class_num = mat['delivery']
fig, axs = plt.subplots(5)
axs[0].plot(data[10000:12000,0])
axs[1].plot(data[10000:12000,1])
axs[2].plot(data[10000:12000,2])
axs[3].plot(data[10000:12000,3])
axs[4].plot(data[10000:12000,4])
plt.show()
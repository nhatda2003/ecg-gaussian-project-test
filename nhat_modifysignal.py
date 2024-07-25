import numpy as np

def generate_gaussian(mean, std_dev):
    value = np.random.normal(mean, std_dev)
    return value

def nhat_modify(signal, pause_idx=0):
    modified = np.zeros(len(signal))
    
    # print(max(signal))
    # print(min(signal))
    
    for i in range(len(signal)):
        
        #current value
        
        
        #Parameters edit
        mean = signal[i]
        #std_dev = abs(signal[i]) * 0.03 #Luu y la std_dev phai la so duong
        #std_dev = abs(signal[i]) * 0.05 #Luu y la std_dev phai la so duong
        std_dev = abs(signal[i]) * 0.07 #Luu y la std_dev phai la so duong
        #std_dev = abs(signal[i]) * 0.09 #Luu y la std_dev phai la so duong
        modified[i] = generate_gaussian(mean, std_dev)
    
    return modified

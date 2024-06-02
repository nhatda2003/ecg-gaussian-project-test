import numpy as np


def generate_gaussian(mean, std_dev):
    """
    Generate a random value from a Gaussian distribution.

    Args:
    - mean (float): Mean of the Gaussian distribution.
    - std_dev (float): Standard deviation of the Gaussian distribution.

    Returns:
    - value (float): Random value generated from the Gaussian distribution.
    """
    value = np.random.normal(mean, std_dev)
    return value
def nhat_modify(signal, pause_idx=0):
    modified = np.zeros(len(signal))
    
    # print(max(signal))
    # print(min(signal))
    # print("----------------------")
    
    for i in range(len(signal)):
        
        #current value
        
        
        #Parameters edit
        mean = signal[i]
        std_dev = abs(signal[i]) * 0.10 #Luu y la std_dev phai la so duong
        modified[i] = generate_gaussian(mean, std_dev)
    
    return modified

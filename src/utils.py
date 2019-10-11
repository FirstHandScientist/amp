'''
This file provide the utility functions for this project.
'''

import numpy as np


def channel_component(hparam):
    return 1/np.sqrt(2*hparam.num_rx)*np.random.randn(hparam.num_rx,hparam.num_tx)

def sampling_signal(hparam):
    x = np.random.choice(hparam.constellation, hparam.num_tx*2, replace=True, p=hparam.soucrce_prior)
    symbol = real2complex(x)
    return x, symbol

def sampling_H(hparam):
    real = channel_component(hparam)
    img = channel_component(hparam)
    real_img = np.concatenate((real, -img), axis=1)
    img_real = np.concatenate((img, real), axis=1)
    return np.concatenate((real_img, img_real), axis=0)

def ERsampling_S(hparam, p):
    S = np.zeros((hparam.num_tx, hparam.num_rx))
    b = np.random.randn(hparam.num_tx)* hparam.stn_var / 4
    
    for i in range(hparam.num_tx):
        
        for j in range(i, hparam.num_rx):
            trial = np.random.random()
            if trial <= p:
                S[i, j] = np.random.randn() * hparam.stn_var
                S[j, i] = S[i, j]
        
        S[i, i] = np.abs(np.random.randn())
        row_max = S[i].max()
        if S[i, i]< row_max:
            S[i, i] = row_max
    
    return (S, b)


def sampling_noise(hparam, snr):
    # noise_var = hparam.num_tx/hparam.num_rx * np.power(10, -snr/10)
    # noise_var = hparam.num_tx * np.power(10, -snr/10)
    noise_var = 1. / snr
    noise = np.sqrt( noise_var) * np.random.randn(hparam.num_rx * 2)
    return (noise, noise_var)

                        
def real2complex(x):
    x = np.array(x)
    num = x.shape[0]
    real = x[:int(num/2)]
    img = x[int(num/2):num]
    return real + 1j * img

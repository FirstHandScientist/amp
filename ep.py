import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
from scipy.stats import multivariate_normal
from modules import GaussianDiag, EP, MMSE
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex
# configuretion
class hparam(object):
    num_tx = 4
    num_rx = 4
    soucrce_prior = [0.2, 0.8]
    signal_var = 1
    snr = np.linspace(1,32,32)
    monte = 5000
    ep_iteration = 16
    constellation = [int(-1), int(1)]

### begin test ########

mmse = MMSE()
ser_mmse = []
ser_ep = []
ser_ep_map = []

for snr in hparam.snr:
    ser_mmse_tmp = []
    ser_ep_tmp = []
    ser_ep_map_tmp = []

    for monte in range(hparam.monte):
        x, true_symbol = sampling_signal(hparam)
        #noise variance in control by SNR in DB
        noise, noise_var = sampling_noise(hparam=hparam, snr=snr)
        channel = sampling_H(hparam)
        noised_signal = np.dot(channel,x) + noise
        
        #### mse detection
        power_ratio = noise_var/hparam.signal_var
        detected_by_mmse = mmse.detect(y=noised_signal, channel=channel, power_ratio=power_ratio)
        mmse_symbol = real2complex(np.sign(detected_by_mmse))
        error_mmse = np.sum(true_symbol != mmse_symbol)
        ser_mmse_tmp.append(error_mmse)

        #### ep detection
        my_ep = EP(noise_var, hparam)
        for i in range(2):
            my_ep.iteration(channel=channel,
                            noise_var=noise_var,
                            noised_signal=noised_signal)
        ep_symbol = real2complex(my_ep.detect_signal_by_mean())
        ser_ep_tmp.append( np.sum(true_symbol != ep_symbol) )

        ep_map_symbol = real2complex( my_ep.detect_signal_by_map() )
        ser_ep_map_tmp.append( np.sum(true_symbol != ep_map_symbol) )

    
    ser_mmse.append( np.mean(ser_mmse_tmp)/hparam.num_tx )
    ser_ep.append( np.mean(ser_ep_tmp)/hparam.num_tx )
    ser_ep_map.append( np.mean(ser_ep_map_tmp)/hparam.num_tx )

    
fig, ax = plt.subplots()
ax.semilogy(hparam.snr, ser_mmse, label = "MMSE")
ax.semilogy(hparam.snr, ser_ep, label = "EP_mean")
ax.semilogy(hparam.snr, ser_ep_map, label = "EP_MAP")
ax.legend()
ax.set(xlabel="SNR", ylabel="SER")
ax.grid()
#fig.savefig("mmse_ep_SER_prior{}.pdf".format(hparam.soucrce_prior))
plt.show()

        
        


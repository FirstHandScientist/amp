import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
from scipy.stats import multivariate_normal
from modules import GaussianDiag, EP, MMSE, PowerEP, StochasticEP, ExpansionEP, ExpansionPowerEP
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex
# configuretion
class hparam(object):
    num_tx = 4
    num_rx = 4
    soucrce_prior = [0.5, 0.5]
    signal_var = 1
    snr = np.linspace(1,32,32)
    monte = 5000
    ep_iteration = 10
    power_n = 4./3
    constellation = [int(-1), int(1)]

    
    #algos_list = ["MMSE", "EP", "PowerEP"]
    algos = {"MMSE": {"detector": MMSE},
             "EP": {"detector": EP},
             "ExpansionEP": {"detector": ExpansionEP},
             "ExpansionPowerEP": {"detector": ExpansionPowerEP}
    }
    for _, value in algos.items():
        value["ser"] = []
    
    


### begin test ########

# mmse = MMSE()
# ser_mmse = []
# ser_ep = []
# ser_ep_map = []
# ser_power_ep = []
# ser_power_ep_map =[]
# ser_sep = []


for snr in hparam.snr:
    tmp = dict()
    for name,_ in hparam.algos.items():
        tmp[name] = []
    # ser_mmse_tmp = []
    # ser_ep_tmp = []
    # ser_ep_map_tmp = []
    # ser_power_ep_tmp = []
    # ser_power_ep_map_tmp = []
    # ser_sep_tmp = []

    for monte in range(hparam.monte):
        x, true_symbol = sampling_signal(hparam)
        #noise variance in control by SNR in DB
        noise, noise_var = sampling_noise(hparam=hparam, snr=snr)
        channel = sampling_H(hparam)
        noised_signal = np.dot(channel,x) + noise
        for key, method in hparam.algos.items():
            if key is "MMSE":
                #### mes detection
                detector = method["detector"]()
                power_ratio = noise_var/hparam.signal_var
                estimated_symbol = detector.detect(y=noised_signal, channel=channel, power_ratio=power_ratio)
                #estimated_symbol = real2complex(np.sign(detected_by_mmse))
            else:
                detector = method['detector'](noise_var, hparam)
                detector.fit(channel=channel,
                          noise_var=noise_var,
                          noised_signal=noised_signal,
                          stop_iter=hparam.ep_iteration)
                estimated_symbol = detector.detect_signal_by_mean()



            est_complex_symbol = real2complex(estimated_symbol)
            error = np.sum(true_symbol != est_complex_symbol)
            
            tmp[key].append(error)

        #### ep detection and performance collections
        # my_ep = EP(noise_var, hparam)
        
        # my_ep.fit(channel=channel,
        #           noise_var=noise_var,
        #           noised_signal=noised_signal,
        #           stop_iter=hparam.ep_iteration)
        # ep_symbol = real2complex(my_ep.detect_signal_by_mean())
        # ser_ep_tmp.append( np.sum(true_symbol != ep_symbol) )

        # ep_map_symbol = real2complex( my_ep.detect_signal_by_map() )
        # ser_ep_map_tmp.append( np.sum(true_symbol != ep_map_symbol) )
        
        #### Stochastic EP detection and performance collections
        # my_sep = StochasticEP(noise_var, hparam)
        # my_sep.fit(channel=channel,
        #           noise_var=noise_var,
        #           noised_signal=noised_signal,
        #           stop_iter=hparam.ep_iteration)
        # sep_symbol = real2complex( my_sep.detect_signal_by_mean() )
        # ser_sep_tmp.append( np.sum(true_symbol != sep_symbol) )
        
        ##### PowerEP detection and performance collections
        # power_EP = PowerEP(noise_var=noise_var,
        #                    hparam=hparam,
        #                    power_n=2)
        # power_EP.fit(channel=channel,
        #              noise_var=noise_var,
        #              noised_signal=noised_signal,
        #              stop_iter=hparam.ep_iteration)
        ## 1st moment detection
        # power_ep_symbol = real2complex(power_EP.detect_signal_by_mean())
        # ser_power_ep_tmp.append( np.sum(true_symbol != power_ep_symbol))
        ## MAP detection
        # power_ep_map_symbol = real2complex( power_EP.detect_signal_by_map() )
        # ser_power_ep_map_tmp.append( np.sum(true_symbol != power_ep_map_symbol))
        
    for key, method in hparam.algos.items():
        method["ser"].append( np.mean(tmp[key])/hparam.num_tx )
    # ser_mmse.append( np.mean(ser_mmse_tmp)/hparam.num_tx )
    # ser_ep.append( np.mean(ser_ep_tmp)/hparam.num_tx )
    #ser_ep_map.append( np.mean(ser_ep_map_tmp)/hparam.num_tx )
    
    ## collection performance of PowerEP of moment detection
    # ser_power_ep.append( np.mean(ser_power_ep_tmp)/hparam.num_tx )
    # ser_power_ep_map.append( np.mean(ser_power_ep_map_tmp)/hparam.num_tx )
    
    ## collection performance of StochasticEP
    # ser_sep.append( np.mean(ser_sep_tmp)/hparam.num_tx )

marker_list = ["o", "<", "+", ">"]
iter_marker_list = iter(marker_list)
fig, ax = plt.subplots()
for key, method in hparam.algos.items():
    ax.semilogy(hparam.snr, method["ser"],
                label = key + "_Iteration:{}".format(hparam.ep_iteration) if "MMSE" not in key else "",
                marker=next(iter_marker_list))
    
# ax.semilogy(hparam.snr, ser_mmse, label = "MMSE")
# ax.semilogy(hparam.snr, ser_ep, label = "EP_mean", marker="o")
# # ax.semilogy(hparam.snr, ser_ep_map, label = "EP_MAP", marker="<")
# ax.semilogy(hparam.snr, ser_power_ep, label = "PowerEP_mean", marker="+")
# ax.semilogy(hparam.snr, ser_power_ep_map, label = "PowerEP_map", marker=">")

# ax.semilogy(hparam.snr, ser_sep, label = "SEP", marker = "1")

ax.legend()
ax.set(xlabel="SNR", ylabel="SER")
ax.grid()
#fig.savefig("mmse_ep_SER_prior{}.pdf".format(hparam.soucrce_prior))
plt.show()

        
        


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
from modules import GaussianDiag, EP, MMSE, PowerEP, StochasticEP, ExpansionEP, ExpansionPowerEP, ExpectationConsistency, LoopyBP, LoopyMP, PPBP
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex

# configuration
class hparam(object):
    num_tx = 4
    num_rx = 4
    soucrce_prior = [0.5, 0.5]
    signal_var = 1
    snr = np.linspace(1,15,15)
    monte = 200
    power_n = 4./3
    constellation = [int(-1), int(1)]

    EC_beta = 0.2
    
    #algos_list = ["MMSE", "EP", "PowerEP"]
    # algos = {"MMSE": {"detector": MMSE},
    #          "EP": {"detector": EP},
    #          "ExpansionEP": {"detector": ExpansionEP},
    #          "ExpansionPowerEP": {"detector": ExpansionPowerEP}
    algos = {"MMSE": {"detector": MMSE},
             "LoopyBP": {"detector": LoopyBP},
             # "LoopyMP": {"detector": LoopyMP},
             "PPBP": {"detector": PPBP}
    }
    iter_num = {"EP": 10,
                "EC": 50,
                "LoopyBP": 10,
                "PPBP": 10,
                "LoopyMP": 10}
    
    for _, value in algos.items():
        value["ser"] = []

def worker(snr):

    tmp = dict()
    for name,_ in hparam.algos.items():
        tmp[name] = []

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
                             stop_iter=hparam.iter_num[key])
                
                        
                estimated_symbol = detector.detect_signal_by_mean()



            est_complex_symbol = real2complex(estimated_symbol)
            error = np.sum(true_symbol != est_complex_symbol)
            
            tmp[key].append(error)

    performance = {"snr": snr}
    for key, method in hparam.algos.items():
        #method["ser"].append( np.mean(tmp[key])/hparam.num_tx )
        performance[key] =  np.mean(tmp[key])/hparam.num_tx 
    return performance

RESULTS = Parallel(n_jobs=1, pre_dispatch="all", verbose=11, backend="threading")(map(delayed(worker), list(hparam.snr)))

performance = defaultdict(list)
for key, _ in hparam.algos.items():
    for the_result in RESULTS:
        performance[key].append( the_result[key] )

    
    
    
# for snr in hparam.snr:


marker_list = ["o", "<", "+", ">"]
iter_marker_list = iter(marker_list)
fig, ax = plt.subplots()
for key, method in hparam.algos.items():
    ax.semilogy(hparam.snr, performance[key],
                # label = key + "_Iteration:{}".format(hparam.iter_num[key]) if "MMSE" not in key else "MMSE",
                label = key,
                marker=next(iter_marker_list))
    

ax.legend()
ax.set(xlabel="SNR", ylabel="SER")
ax.grid()
fig.savefig("figures/experiments_3-25.pdf")
plt.show()

        
        


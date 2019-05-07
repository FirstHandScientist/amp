import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
from modules import GaussianDiag, EP, MMSE, PowerEP, StochasticEP, ExpansionEP, ExpansionPowerEP, ExpectationConsistency, LoopyBP, LoopyMP, PPBP, AlphaBP, MMSEalphaBP, ML
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex



# configuration
class hparam(object):
    num_tx = 4
    num_rx = 4
    soucrce_prior = [0.5, 0.5]
    signal_var = 1
    snr = np.linspace(1,15,15)
    monte = 20
    power_n = 4./3
    constellation = [int(-1), int(1)]

    EC_beta = 0.2
    alpha = 0.7
    #algos_list = ["MMSE", "EP", "PowerEP"]
    # algos = {"MMSE": {"detector": MMSE},
    #          "EP": {"detector": EP},
    #          "ExpansionEP": {"detector": ExpansionEP},
    #          "ExpansionPowerEP": {"detector": ExpansionPowerEP}
    algos = {"MMSE": {"detector": MMSE},
             "ML": {"detector": ML},
             "LoopyBP": {"detector": LoopyBP},
             #"LoopyMP": {"detector": LoopyMP},
             "AlphaBP": {"detector": AlphaBP},
             "MMSEalphaBP": {"detector": MMSEalphaBP},
             "PPBP": {"detector": PPBP}
    }
    iter_num = {"EP": 10,
                "EC": 50,
                "LoopyBP": 50,
                "PPBP": 50,
                "AlphaBP": 50,
                "MMSEalphaBP": 50,
                "LoopyMP": 50}
    
    for _, value in algos.items():
        value["ser"] = []

def task(snr):

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
            if key is "MMSE" or key is "ML":
                #### mes detection
                detector = method["detector"](hparam)
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

results = []
def collect_result(result):
    global results
    results.append(result)

pool = mp.Pool(mp.cpu_count())

# RESULTS = Parallel(n_jobs=1, pre_dispatch="all", verbose=11, backend="threading")(map(delayed(worker), list(hparam.snr)))
# for snr in list(hparam.snr):
#     pool.apply_async(task, args=(snr), callback=collect_result)
results = pool.map(task, list(hparam.snr))


#results = [r for r in result_objects]

pool.close()


performance = defaultdict(list)

#for the_result in RESULTS:
for snr in list(hparam.snr):
    for the_result in results:
        if the_result["snr"] == snr:
            for key, _ in hparam.algos.items():                
                performance[key].append( the_result[key] )

    
    
    
# for snr in hparam.snr:


marker_list = ["o", "<", "+", ">", "v", "1", "2", "3", "8"]
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
fig.savefig("figures/experiments_5-7.pdf")
plt.show()

        
        


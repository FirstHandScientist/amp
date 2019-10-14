# coding: utf-8

'''
Experiments to produce results in Figure 3b in amp.pdf, MIMO detection: α-BP with prior

The comparison between algorithms alpha-BP using prior of mmse with different alpha value, mmse, MAP
'''
# package dependencies
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
import sys

# import algorithms
sys.path.append("./src")
from modules import GaussianDiag, EP, MMSE, PowerEP, StochasticEP, ExpansionEP, ExpansionPowerEP, ExpectationConsistency, LoopyBP, LoopyMP, PPBP, AlphaBP, MMSEalphaBP, ML, VariationalBP, MMSEvarBP, EPalphaBP
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex


# configuration of experiments
class hparam(object):
    # vector length of x = num_tx *2, section 1 in amp.pdf
    num_tx = 4
    # vector length of y = num_rx *2, section 1 in amp.pdf
    num_rx = 4
    constellation = [int(-1), int(1)]
    soucrce_prior = [0.5, 0.5]
    signal_var = 1
    snr = np.linspace(1, 40, 10)
    # number of monte carlo simulations per point in the experiment figure
    monte = 5
    power_n = 4./3
    EC_beta = 0.2
    alpha = None
    ########### import test with pysudo prior ################
    algos = {"MMSE": {"detector": MMSE, "legend": "MMSE"},
             "ML": {"detector": ML, "legend": "MAP"},
             "LoopyBP": {"detector": LoopyBP, "legend": "BP"},
             "MMSEalphaBP, 0.2": {"detector": MMSEalphaBP, "alpha": 0.2, "legend":r'$\alpha$-BP+MMSE, 0.2'},
             "MMSEalphaBP, 0.4": {"detector": MMSEalphaBP, "alpha": 0.4, "legend":r'$\alpha$-BP+MMSE, 0.4'},
             "MMSEalphaBP, 0.6": {"detector": MMSEalphaBP, "alpha": 0.6, "legend":r'$\alpha$-BP+MMSE, 0.6'},
             "MMSEalphaBP, 0.8": {"detector": MMSEalphaBP, "alpha": 0.8, "legend":r'$\alpha$-BP+MMSE, 0.8'},

    }
    
    iter_num = {"EP": 10,
                "EC": 50,
                "LoopyBP": 50,
                "PPBP": 50,
                "AlphaBP": 50,
                "MMSEalphaBP": 50,
                "VariationalBP":50,
                "EPalphaBP": 50,
                "MMSEvarBP":50,
                "LoopyMP": 50}
    
    for _, value in algos.items():
        value["ser"] = []

def task(snr):
    '''
    Given the snr value, do the experiment with setting defined in hparam
    '''
    tmp = dict()
    for name,_ in hparam.algos.items():
        tmp[name] = []

    for monte in tqdm(range(hparam.monte)):
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
                if "Alpha" in key or "alpha" in key:
                    hparam.alpha = method['alpha']

                detector = method['detector'](noise_var, hparam)
                detector.fit(channel=channel,
                             noise_var=noise_var,
                             noised_signal=noised_signal,
                             stop_iter=50)
                
                        
                estimated_symbol = detector.detect_signal_by_mean()



            est_complex_symbol = real2complex(estimated_symbol)
            error = np.sum(true_symbol != est_complex_symbol)
            
            tmp[key].append(error)

    performance = {"snr": snr}
    for key, method in hparam.algos.items():
        #method["ser"].append( np.mean(tmp[key])/hparam.num_tx )
        performance[key] =  np.mean(np.array(tmp[key]))/hparam.num_tx

 
    return performance

# begin the experiment
if (__name__ == '__main__'):
    results = []
    def collect_result(result):
        global results
        results.append(result)

    pool = mp.Pool(mp.cpu_count())
    results = pool.map(task, list(hparam.snr))
    pool.close()

    performance = defaultdict(list)

    #for the_result in RESULTS:
    for snr in list(hparam.snr):
        for the_result in results:
            if the_result["snr"] == snr:
                for key, _ in hparam.algos.items():                
                    performance[key].append( the_result[key] )


    # plot the experiments results
    marker_list = ["o", "<", "+", ">", "v", "1", "2", "3", "8", "*", "h", "d", "D"]
    iter_marker_list = iter(marker_list)
    fig, ax = plt.subplots()
    for key, method in hparam.algos.items():
        ax.semilogy(hparam.snr, performance[key],
                    label = method['legend'],
                    marker=next(iter_marker_list))

    lgd = ax.legend(bbox_to_anchor=(1.64,1), borderaxespad=0)
    ax.set(xlabel="Ratio of Signal to Noise Variance", ylabel="SER")
    ax.grid()
    fig.savefig("figures/prior_mmse_alpha_compare.pdf",
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    #plt.show()

        
        


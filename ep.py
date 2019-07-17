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
import pickle
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
from modules import GaussianDiag, EP, MMSE, PowerEP, StochasticEP, ExpansionEP, ExpansionPowerEP, ExpectationConsistency, LoopyBP, LoopyMP, PPBP, AlphaBP, MMSEalphaBP, ML, VariationalBP, MMSEvarBP, EPalphaBP, StochasticBP
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex


# define the progress bar to show the progress


# configuration
class hparam(object):
    num_tx = 2
    num_rx = 2
    soucrce_prior = [0.25, 0.25, 0.25, 0.25]
    signal_var = 1
    snr = np.linspace(1,100, 10)
    monte = 30
    constellation = [int(-2), int(-1), int(1), int(2)]
    alpha = 0.4
    power_n = 2
    EC_beta = 0.2
    algos = {"MMSE": {"detector": MMSE, "legend": "MMSE"},
             "ML": {"detector": ML, "legend": "MAP"},
             "PowerEP": {"detector": PowerEP, "legend": "Power EP"},
             "StochasticEP": {"detector": StochasticEP, "legend": "StochasticEP"},
             "EP": {"detector": EP, "legend": "EP"},
             "EC": {"detector": ExpectationConsistency, "legend": "EC"},
             # "AlphaBP": {"detector": AlphaBP, "legend": r'$\alpha$-BP,'+' {}'.format(alpha)},
             "MMSEalphaBP": {"detector": MMSEalphaBP, "legend":r'$\alpha$-BP+MMSE,'+' {}'.format(alpha)},
             "ExpansionEP": {"detector": ExpansionEP, "legend": "Expansion EP"},
             "StochasticBP": {"detector": StochasticBP, "legend":"Stochastic BP,"+' {}'.format(alpha)},
             
             "EPalphaBP": {"detector": EPalphaBP, "legend": r'$\alpha$-BP+EP,'+' {}'.format(alpha)}
             }
    
    iter_num = {"EP": 10,
                "EC": 10,
                "PowerEP": 10,
                "StochasticEP":10,
                "ExpansionEP": 10,
                "StochasticBP":50,
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


#pbar = tqdm(total=len(list(hparam.snr)))

def task(snr):

    tmp = dict()
    for name,_ in hparam.algos.items():
        tmp[name] = []

    #progress = tqdm(range(hparam.monte))
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
                detector = method['detector'](noise_var, hparam)
                detector.fit(channel=channel,
                             noise_var=noise_var,
                             noised_signal=noised_signal,
                             stop_iter=hparam.iter_num[key])
                
                        
                estimated_symbol = detector.detect_signal_by_mean()



            # est_complex_symbol = real2complex(estimated_symbol)
            error = np.sum(x != estimated_symbol)
            
            tmp[key].append(error)

    performance = {"snr": snr}
    for key, method in hparam.algos.items():
        #method["ser"].append( np.mean(tmp[key])/hparam.num_tx )
        performance[key] =  np.mean(np.array(tmp[key]))/hparam.num_tx

 
    return performance

if (__name__ == '__main__'):
    results = []
    def collect_result(result):
        global results
        results.append(result)

    pool = mp.Pool(mp.cpu_count())
    # pool = mp.Pool(hparam.snr.shape[0])

    # RESULTS = Parallel(n_jobs=1, pre_dispatch="all", verbose=11, backend="threading")(map(delayed(worker), list(hparam.snr)))
    # for snr in list(hparam.snr):
    #     pool.apply_async(task, args=(snr), callback=collect_result)
    # task(hparam.snr[1])
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




    # save the experiments results first
    with open("figures/ep_results.pkl", 'wb') as handle:
        pickle.dump(performance, handle)


    # Plot the experiments results
    marker_list = ["o", "<", "+", ">", "v", "1", "2", "3", "8", "*", "h", "d", "D"]
    iter_marker_list = iter(marker_list)
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    for key, method in hparam.algos.items():
        ax.semilogy(hparam.snr, performance[key],
                    label = method['legend'],
                    marker=next(iter_marker_list))

    lgd = ax.legend(bbox_to_anchor=(1.64,1), borderaxespad=0)

    ax.set(xlabel="ration of signal to noise variance", ylabel="SER")
    ax.grid()
    fig.savefig("figures/ep_experiments_alpha{}.pdf".format(int(hparam.alpha/0.1)), bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show()

        
        


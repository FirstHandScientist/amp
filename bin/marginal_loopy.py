import numpy as np
import pickle
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib
# matplotlib.rcParams.update({'font.size': 18})
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
# from modules import GaussianDiag, EP, MMSE, PowerEP, StochasticEP, ExpansionEP, ExpansionPowerEP, ExpectationConsistency, LoopyBP, LoopyMP, PPBP, AlphaBP, MMSEalphaBP, ML, VariationalBP, MMSEvarBP, EPalphaBP
import sys
sys.path.append("./src")

from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex


import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
from loopy_modules import LoopyBP, AlphaBP, ML, MMSEalphaBP, Marginal
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex, ERsampling_S



# configuration
class hparam(object):
    num_tx = 9
    num_rx = 9
    soucrce_prior = [0.5, 0.5]
    signal_var = 1
    connect_prob = np.linspace(0.0, 0.9, 10)
    
    monte = 30
    constellation = [int(-1), int(1)]

    alpha = None
    stn_var= 1
    # algos = {"LoopyBP": {"detector": LoopyBP, "alpha": None},
    # }
    
    algos = {"BP": {"detector": LoopyBP, "alpha": None, "legend": "BP", "row": 0}, 
             # "AlphaBP, 0.2": {"detector": AlphaBP, "alpha": 0.2, "legend": r'$\alpha$-BP, 0.2'},
             # "MMSEalphaBP, 0.4": {"detector": MMSEalphaBP, "alpha": 0.4, "legend": r'$\alpha$-BP+MMSE, 0.4', "row": 1},
             "AlphaBP, 0.4": {"detector": AlphaBP, "alpha": 0.4, "legend": r'$\alpha$-BP, 0.4', "row": 1},
             # "AlphaBP, 0.6": {"detector": AlphaBP, "alpha": 0.6, "legend": r'$\alpha$-BP, 0.6'},
             "AlphaBP, 0.8": {"detector": AlphaBP, "alpha": 0.8, "legend": r'$\alpha$-BP, 0.8',"row": 2},
             "AlphaBP, 1.2": {"detector": AlphaBP, "alpha": 1.2, "legend": r'$\alpha$-BP, 1.2', "row": 3}

    }
    iter_num = 100

    
    for _, value in algos.items():
        value["ser"] = []

def task(erp):

    tmp = dict()
    for name,_ in hparam.algos.items():
        tmp[name] = []

    for key, method in hparam.algos.items():
        dict_marg = {"true":[], "est": []}
        for monte in range(hparam.monte):
            # sampling the S and b for exponential function
            S, b = ERsampling_S(hparam, erp)

            # compute the joint ML detection
            detectMg = Marginal(hparam)
            true_marginals = detectMg.detect(S, b)

            # marginals estimated
            hparam.alpha = method['alpha']
            detector = method['detector'](None, hparam)
            detector.fit(S=S,
                         b=b,
                         stop_iter=hparam.iter_num)


            estimated_marginals = detector.marginals()
            ## concatenate the marginals
            dict_marg["true"] = np.concatenate((dict_marg["true"], true_marginals[:,0]))
            dict_marg["est"] = np.concatenate((dict_marg["est"], estimated_marginals[:,0]))


        

        tmp[key].append(dict_marg)

    # performance should be made by comparing with ML
    performance = {"erp": erp}
    for key, method in hparam.algos.items():
        #method["ser"].append( np.mean(tmp[key])/hparam.num_tx )
        performance[key] =  tmp[key]
    return performance

results = []
def collect_result(result):
    global results
    results.append(result)


# task(hparam.connect_prob[0])
pool = mp.Pool(mp.cpu_count())

results = pool.map(task, list(hparam.connect_prob))

pool.close()


performance = defaultdict(list)

#for the_result in RESULTS:
for connect_prob in list(hparam.connect_prob):
    for the_result in results:
        if the_result["erp"] == connect_prob:
            for key, _ in hparam.algos.items():                
                performance[key].append( the_result[key] )

    
# save the experimental results    
with open("figures/marginal_prob.pkl", 'wb') as handle:
    pickle.dump(performance, handle)
    
# for snr in hparam.snr:


marker_list = ["o", "<", "+", ">", "v", "1", "2", "3", "8"]
iter_marker_list = iter(marker_list)

figure_format = 2*100 + hparam.connect_prob * 10
figure = plt.figure(figsize=(30,30))
for key, method in hparam.algos.items():
    for i, prob in enumerate(hparam.connect_prob):
        ax = figure.add_subplot(len(hparam.algos), 10, i+1 + method["row"] * 10, adjustable='box', aspect=1)
        ax.scatter(performance[key][i][0]["true"],
                   performance[key][i][0]["est"])
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        
    


# .set(xlabel="Edge Probability", ylabel="MAP Accuracy")
# ax.grid()
figure.tight_layout()
figure.savefig("figures/marginal_acc.pdf")
figure.show()

        
        


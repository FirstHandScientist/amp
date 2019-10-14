'''This file is used to plot the experiment only by loading experiment results'''

import numpy as np
import pickle
import sys
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ep import hparam

if (__name__ == '__main__'):
    usage = "python bin/plot_save.py path/to/figure/date"
    
    fig_file = sys.argv[1]
    hparam = hparam()
    with open(fig_file, 'rb') as handle:
        performance = pickle.load(handle)

    # Plot the experiments results
    marker_list = ["o", "<", "+", ">", "v", "1", "2", "3", "8", "*", "h", "d", "D"]
    iter_marker_list = iter(marker_list)
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    for key, method in hparam.algos.items():
        ax.semilogy(hparam.snr, np.array(performance[key]),
                    label = method['legend'],
                    marker=next(iter_marker_list))

    lgd = ax.legend(bbox_to_anchor=(1.64,1), borderaxespad=0)

    ax.set(xlabel="ration of signal to noise variance", ylabel="SER")
    ax.grid()
    ax.set_ylim([5e-1, 1])
    fig.savefig("figures/ep_experiments_alpha{}_.pdf".format(int(hparam.alpha/0.1)), bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show()

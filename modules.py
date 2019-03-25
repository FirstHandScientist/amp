import numpy as np
import itertools
from scipy.stats import multivariate_normal


######################################################################
class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        return -0.5 * (logs * 2. + ((x - mean) ** 2) / np.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return likelihood

class EP(object):
    
    def __init__(self, noise_var, hparam):
        #### two intermediate parameters needed to update
        self.gamma = np.zeros(hparam.num_tx*2)
        self.Sigma =  np.ones(hparam.num_tx*2) / hparam.signal_var
        #### two parameters for q(u)
        self.mu = np.zeros_like(self.gamma)
        self.covariance = np.diag(np.ones_like(self.gamma))
        #### two parameters for cavity 
        self.cavity_mgnl_h = np.zeros_like(self.gamma)
        self.cavity_mgnl_t = np.zeros_like(self.gamma)
        #### two prox distribution parameters
        self.prox_mu = np.zeros_like(self.gamma)
        self.prox_var = np.zeros_like(self.gamma)

        self.constellation = hparam.constellation

    def get_moments(self):
        return (self.mu, self.covariance)
    

    def update_moments(self, channel, noise_var, noised_signal):
        noised_signal = np.array(noised_signal)
        self.covariance = np.linalg.inv(np.matmul(channel.T, channel)/noise_var
                                        + np.diag(self.Sigma))
        tmp = channel.T.dot(noised_signal)/noise_var
        self.mu = np.dot(self.covariance,
                         tmp+ self.gamma )
        #assert np.all(self.covariance>=0)

    def update_cavity(self):
        for i in range(self.mu.shape[0]):
            self.cavity_mgnl_h[i] = self.covariance[i, i]/( 1 - self.covariance[i, i] * self.Sigma[i])
            self.cavity_mgnl_t[i] = self.cavity_mgnl_h[i] * (self.mu[i]/self.covariance[i,i] - self.gamma[i])
            assert np.all(self.cavity_mgnl_h>=0)


    def update_prox_moments(self):
        vary_small = 1e-6
        #gaussian = GaussianDiag()
        mean = self.cavity_mgnl_t
        #logs = np.log(self.cavity_mgnl_h)/2
        z = None
        for i, the_mean in enumerate(mean):
            # logp = gaussian.likelihood(mean=the_mean,
            #                            logs=logs[i],
            #                            x= np.array(self.constellation))
            logp = np.log(multivariate_normal.pdf(x=np.array(self.constellation),
                                                  mean=the_mean,
                                                  cov=self.cavity_mgnl_h[i]) + vary_small )
            z = np.sum(np.exp(logp))
            self.prox_mu[i] = np.array(self.constellation).dot( np.exp(logp) )/z
            # second_moment = np.power(np.array(self.constellation), 2).dot( np.exp(logp) )/z
            # self.prox_var[i] = second_moment -  np.power(self.prox_mu[i], 2)
            self.prox_var[i] = np.power(np.array(self.constellation) - self.prox_mu[i], 2).dot( np.exp(logp) )/z
            assert np.all(self.prox_var>=0)

    def kl_match_momoents(self):
        vary_small = 1e-6
        Sigma = 1./(self.prox_var + vary_small) - 1./(self.cavity_mgnl_h + vary_small)
        gamma = self.prox_mu / (self.prox_var+vary_small) - self.cavity_mgnl_t / (self.cavity_mgnl_h+vary_small)
        if np.any(np.isnan(Sigma)) and np.any(np.isnan(gamma)):
            print("value error")
        if np.any(Sigma>0):
            positive_idx = Sigma>0
            self.Sigma[positive_idx] = Sigma[positive_idx]
            self.gamma[positive_idx] = gamma[positive_idx]
            assert np.all(self.Sigma>0)

    def fit(self, channel, noise_var, noised_signal, stop_iter=10):
        """Do the training by number of iteration of stop_iter"""
        for i in range(stop_iter):
            self.update_moments(channel=channel,
                                noise_var=noise_var,
                                noised_signal=noised_signal)
            self.update_cavity()
            self.update_prox_moments()
            self.kl_match_momoents()


    def detect_signal_by_mean(self):
        estimated_signal = []
        for mu in self.mu:
            obj_list  = np.abs(mu - np.array(self.constellation))
            estimated_signal.append(self.constellation[np.argmin(obj_list)])
        return estimated_signal

    
    def detect_signal_by_map(self):
        
        mean = self.mu
        cov = self.covariance
        proposals = list( itertools.product(self.constellation, repeat=mean.shape[0]) )
        
        p_list = multivariate_normal.pdf(x=proposals, mean=mean, cov=cov) 

        # for x in proposals:
        #     logp = np.log(multivariate_normal.pdf(x=x, mean=mean, cov=cov) )
        #     logp_list.append(logp)
        idx_max = np.argmax( p_list )
        return proposals[idx_max]

class PowerEP(EP):
    
    def __init__(self, noise_var, hparam, power_n=1):
        super().__init__(noise_var, hparam)
        #### set the power n for PowerEP, integer
        self.power_n = hparam.power_n
        #assert self.power_n > 0

    def update_cavity(self):
        for i in range(self.mu.shape[0]):
            self.cavity_mgnl_h[i] = self.covariance[i, i]/( 1 - self.covariance[i, i] * self.Sigma[i] / self.power_n)
            self.cavity_mgnl_t[i] = self.cavity_mgnl_h[i] * (self.mu[i]/self.covariance[i,i] - self.gamma[i] / self.power_n)

        assert np.all(self.cavity_mgnl_h>=0)
        
    def kl_match_momoents(self):
        vary_small = 1e-6
        Sigma = self.power_n * (1./(self.prox_var + vary_small) - 1./(self.cavity_mgnl_h + vary_small))
        gamma = self.power_n * (self.prox_mu / (self.prox_var+vary_small) - self.cavity_mgnl_t / (self.cavity_mgnl_h+vary_small))
        if np.any(np.isnan(Sigma)) and np.any(np.isnan(gamma)):
            print("value error")
        if np.any(Sigma>0):
            positive_idx = Sigma>0
            self.Sigma[positive_idx] = Sigma[positive_idx]
            self.gamma[positive_idx] = gamma[positive_idx]
        assert np.all(self.Sigma>0)

### definition of stochastic EP:
class StochasticEP(EP):
    def __init__(self, noise_var, hparam):
        super().__init__(noise_var, hparam)

    def kl_match_momoents(self):
        """Moment matching step"""
        vary_small = 1e-6
        Sigma = 1./(self.prox_var + vary_small) - 1./(self.cavity_mgnl_h + vary_small)
        gamma = self.prox_mu / (self.prox_var+vary_small) - self.cavity_mgnl_t / (self.cavity_mgnl_h+vary_small)
        if np.any(np.isnan(Sigma)) and np.any(np.isnan(gamma)):
            print("value error")
        if np.any(Sigma>0):
            positive_idx = Sigma>0
            n = np.sum(positive_idx)
            total_n = self.Sigma.size
            self.Sigma = ( self.Sigma[0] * (1 - n / total_n) +  np.sum(Sigma[positive_idx]) * ( 1 / total_n) ) * np.ones_like(self.Sigma)
            self.gamma = (self.gamma[0] * ( 1 - n / total_n) +  np.sum(gamma[positive_idx]) * ( 1 / total_n) ) * np.ones_like(self.gamma)
            assert np.all(self.Sigma>0)
    

class ExpansionEP(EP):
    """Do the improvement of EP, using the basic expansion: the p(x) is appx as 
    \sum_n q_n(x) - (N-1) p(x), q_n is the n-th title distribution"""
    
    def __init__(self, noise_var, hparam):
        super().__init__(noise_var, hparam)
        
    def detect_signal_by_mean(self):
        """1st order correction"""
        estimated_signal = []
        
        for mu in self.prox_mu:
            obj_list  = np.abs(mu - np.array(self.constellation))
            estimated_signal.append(self.constellation[np.argmin(obj_list)])
        return estimated_signal

class ExpansionPowerEP(PowerEP):
    """Do the improvement of EP, using the basic expansion: the p(x) is appx as 
    \sum_n q_n(x) - (N-1) p(x), q_n is the n-th title distribution"""
    
    def __init__(self, noise_var, hparam):
        super().__init__(noise_var, hparam)
        
    def detect_signal_by_mean(self):
        """1st order correction to power EP"""
        
        estimated_signal = []
        
        for mu in self.prox_mu:
            obj_list  = np.abs(mu - np.array(self.constellation))
            estimated_signal.append(self.constellation[np.argmin(obj_list)])
        return estimated_signal


class MMSE(object):
    def __init__(self):
        pass
    def detect(self, y, channel, power_ratio):
        inv = np.linalg.inv(power_ratio * np.eye(channel.shape[1]) 
                            + np.matmul(channel.T, channel) )
        x = inv.dot(channel.T).dot(y)
        return np.sign(x)


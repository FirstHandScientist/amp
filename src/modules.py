import numpy as np
import itertools
import factorgraph as fg
import scipy.sparse.csgraph as csgraph
import maxsum
import alphaBP
import variationalBP
from scipy.stats import multivariate_normal


######################################################################
class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X -n Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
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
        super(PowerEP, self).__init__(noise_var, hparam)
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
            assert np.all(self.Sigma>=0)
    

class ExpansionEP(EP):
    """Do the improvement of EP, using the basic expansion: the p(x) is appx as 
    \sum_n q_n(x) - (N-1) p(x), q_n is the n-th title distribution"""
    
    # def __init__(self, noise_var, hparam):
    #     super().__init__(noise_var, hparam)
        
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

    
class ExpectationConsistency(object):
    """The implementation of EC algorithm"""
    vary_small = 1e-6

    def __init__(self, noise_var, hparam):
        self.gamma_q = np.zeros(hparam.num_tx*2)
        self.Sigma_q =  np.ones(hparam.num_tx*2) / hparam.signal_var
        self.gamma_r = np.zeros(hparam.num_tx*2)
        self.Sigma_r =  np.ones(hparam.num_tx*2) / hparam.signal_var
        self.gamma_s = np.zeros(hparam.num_tx*2)
        self.Sigma_s =  np.ones(hparam.num_tx*2) / hparam.signal_var

        self.constellation = np.array(hparam.constellation)
        self.EC_beta = hparam.EC_beta
        self.mu = np.zeros(hparam.num_tx*2)

        self.global_iter_num = 0

    def solve_for_s(self, moment1, moment2):
        """Solve for the parameters of s given moments"""
        inverse_Sigma_s = moment2 - np.power(moment1, 2)
        Sigma_s = 1 / (inverse_Sigma_s + ExpectationConsistency.vary_small)
        #assert np.all(Sigma_s>=0), "Second moment of s should be positive."
        Sigma_s = np.clip(Sigma_s, a_min=1e-3, a_max=1e2)
        gamma_s = Sigma_s * moment1
        
        try:
            assert np.all(np.logical_not(np.isnan(gamma_s))) and np.all(np.logical_not(np.isnan(Sigma_s)))
        except:
            print("Invalid update encountered...")
        
        return gamma_s, Sigma_s

    def update_moments_q(self, channel, noise_var, noised_signal):
        """Update the 1st and 2ed moments of q"""
        noised_signal = np.array(noised_signal)
        g = channel.T.dot(noised_signal)/noise_var + self.gamma_q
        S = channel.T.dot(channel)/noise_var + np.diag(self.Sigma_q)

        covariance = np.linalg.inv(S)
        
        moment1 = covariance.dot(g)
        #self.mu_q = moment1
        moment2 = np.diag(covariance) + np.power( moment1, 2)
                
        try:
            assert np.all(moment2>=0), "Second moment of q should be positive."
        except:
            print("Encounter negative moment2: {}".format(moment2))

        assert np.all(np.logical_not(np.isnan(moment1))) and np.all(np.logical_not(np.isnan(moment2)))
        return moment1, moment2

    def update_moments_r(self):
        """Update the 1st and 2ed moments of r"""
        denominator = np.exp(self.gamma_r[:, None] * self.constellation
                             - self.Sigma_r[:, None] * np.power(self.constellation, 2) /2 )
        nominator1 = np.exp(self.gamma_r[:, None] * self.constellation
                             - self.Sigma_r[:, None] * np.power(self.constellation, 2) /2 ) * self.constellation
        
        nominator2 = np.exp(self.gamma_r[:, None] * self.constellation
                             - self.Sigma_r[:, None] * np.power(self.constellation, 2) /2) * np.power(self.constellation, 2)
        try:
            
            moment1 = nominator1.sum(axis=1) / denominator.sum(axis=1)
            moment2 = nominator2.sum(axis=1) / denominator.sum(axis=1)
            assert np.all(np.logical_not(np.isnan(moment1))) and np.all(np.logical_not(np.isnan(moment2)))
        except:
            print("Oops!  That was no valid number.  Try again...")

        
        self.mu = moment1
        return moment1, moment2

    def get_parameter_s_from_q(self, channel, noise_var, noised_signal):
        """Compute the parameters of s, given moments of q"""
        moment1_q, moment2_q = self.update_moments_q(channel, noise_var, noised_signal)
        gamma_s, Sigma_s = self.solve_for_s(moment1=moment1_q,
                                            moment2=moment2_q)
        self.gamma_s = gamma_s
        self.Sigma_s = Sigma_s
        
    
    def get_parameter_s_from_r(self, channel, noise_var, noised_signal):
        """Compute the parameters of s, given moments of r"""
        moment1_r, moment2_r = self.update_moments_r()

        clip_moment2 = np.max([np.power(moment1_r, 2) + np.power(2., - np.max([1, self.global_iter_num -4]))
                               , moment2_r])
        #clip_moment2 = moment2_r

        gamma_s, Sigma_s = self.solve_for_s(moment1=moment1_r,
                                            moment2=clip_moment2)
        self.gamma_s = gamma_s
        self.Sigma_s = Sigma_s
        

    def update_r(self):
        """Update the parameters of distribution r"""
        self.gamma_r = self.gamma_s - self.gamma_q
        self.Sigma_r = self.Sigma_s - self.Sigma_q
        

    def update_q(self):
        """Update the parameters of distribution q"""
        beta = self.EC_beta
        self.gamma_q = (self.gamma_s - self.gamma_r) * beta + (1 - beta) * self.gamma_q
        self.Sigma_q = (self.Sigma_s - self.Sigma_r) * beta + (1 - beta) * self.Sigma_q
        try:
            assert np.all(np.logical_not(np.isnan(self.gamma_q)))
        except:
            print("Invalid update encountered...")
        
    def fit(self, channel, noise_var, noised_signal, stop_iter=10):
        """Do the training by number of iteration of stop_iter"""
        for i in range(stop_iter):
            self.global_iter_num = i
            
            self.get_parameter_s_from_q(channel, noise_var, noised_signal)
            self.update_r()
            self.get_parameter_s_from_r(channel, noise_var, noised_signal)
            self.update_q()

    def detect_signal_by_mean(self):
        
        # diff_abs = np.abs(self.mu_q[:, None] - self.constellation)
        # estimated_idx = np.argmin(diff_abs, axis=1)
        estimated_signal = []
        for mu in self.mu:
            obj_list  = np.abs(mu - np.array(self.constellation))
            estimated_signal.append(self.constellation[np.argmin(obj_list)])
        return estimated_signal


            
class MMSE(object):
    def __init__(self, hparam):
        self.constellation = hparam.constellation
        
    def detect(self, y, channel, power_ratio):
        inv = np.linalg.inv(power_ratio * np.eye(channel.shape[1]) 
                            + np.matmul(channel.T, channel) )
        x = inv.dot(channel.T).dot(y)
        
        estimated_x = [self.constellation[np.argmin(np.abs(x_i - np.array(self.constellation)))] for x_i in x]
        return np.array(estimated_x)


class ML(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.constellation = hparam.constellation
        pass
    
    def detect(self, y, channel, power_ratio):
                
        proposals = list( itertools.product(self.constellation, repeat=channel.shape[1]) )

        threshold = np.inf
        solution = None
        for x in proposals:
            tmp = np.array(channel).dot(x[:]) - y
            if np.dot(tmp, tmp) < threshold:
                threshold = tmp.T.dot(tmp)
                solution = x

        return solution



class LoopyBP(object):

    def __init__(self, noise_var, hparam):
        # get the constellation
        self.constellation = hparam.constellation
        self.hparam = hparam
        # set the graph
        self.graph = fg.Graph()
        # add the discrete random variables to graph
        self.n_symbol = hparam.num_tx * 2
        for idx in range(hparam.num_tx * 2):
            self.graph.rv("x{}".format(idx), len(self.constellation))

    def set_potential(self, h_matrix, observation, noise_var):
        s = np.matmul(h_matrix.T, h_matrix)
        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_potential = (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2) + h_matrix[:, var_idx].dot(observation) * np.array(self.constellation))/noise_var
            f_potential = f_potential - f_potential.max()
            f_x_i = np.exp(f_potential )
            self.graph.factor(["x{}".format(var_idx)],
                              potential=f_x_i)


        for var_idx in range(h_matrix.shape[1]):

            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                t_potential = - np.array(self.constellation)[None,:].T * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var
                t_potential = t_potential - t_potential.max()
                t_ij = np.exp(t_potential)
                self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                  potential=t_ij)
        

    
    def fit(self, channel, noise_var, noised_signal, stop_iter=10):
        """ set potentials and run message passing"""
        self.set_potential(h_matrix=channel,
                           observation=noised_signal,
                           noise_var=noise_var)

        # run BP
        iters, converged = self.graph.lbp(normalize=True)
        
        
    def detect_signal_by_mean(self):
        estimated_signal = []
        rv_marginals = dict(self.graph.rv_marginals())
        for idx in range(self.n_symbol):
            x_marginal = rv_marginals["x{}".format(idx)]
            
            estimated_signal.append(self.constellation[x_marginal.argmax()])
        return estimated_signal
    
class AlphaBP(LoopyBP):
    def __init__(self, noise_var, hparam):
        self.hparam = hparam
        # get the constellation
        self.constellation = hparam.constellation

        self.n_symbol = hparam.num_tx * 2
        # set the graph
        self.graph = alphaBP.alphaGraph(alpha=hparam.alpha)
        # add the discrete random variables to graph
        for idx in range(hparam.num_tx * 2):
            self.graph.rv("x{}".format(idx), len(self.constellation))


class StochasticBP(AlphaBP):
    def __init__(self, noise_var, hparam):
        self.hparam = hparam
        # get the constellation
        self.constellation = hparam.constellation
        self.alpha = hparam.alpha
        self.n_symbol = hparam.num_tx * 2
        # set the graph
        self.learning_rate = 1
        self.first_iter_flag = True


    def subgraph_mask(self, size):
        """give the mask for spanning tree subgraph"""
        init_matrix = np.random.randn(size,size)
        Tcs = csgraph.minimum_spanning_tree(init_matrix)
        mask_matrix = Tcs.toarray()
        return mask_matrix

    def new_graph(self, h_matrix, observation, noise_var):
        # initialize new graph
        subgraph = alphaBP.alphaGraph(alpha=self.alpha)
        # add the discrete random variables to graph
        for idx in range(h_matrix.shape[1]):
            subgraph.rv("x{}".format(idx), len(self.constellation))

        s = np.matmul(h_matrix.T, h_matrix)

        # get the prior belief
        if not self.first_iter_flag:
            rv_marginals = dict(self.graph.rv_marginals())

        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_potential = (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2)
                             + h_matrix[:, var_idx].dot(observation) * np.array(self.constellation))/noise_var
            f_potential = f_potential - f_potential.max()
            f_x_i = np.exp( f_potential)
            f_x_i = f_x_i/f_x_i.sum()
            if not self.first_iter_flag:
                old_prior = rv_marginals["x{}".format(var_idx)]
                subgraph.factor(["x{}".format(var_idx)],
                                potential=np.power(f_x_i, self.learning_rate) * old_prior)
            else:
                subgraph.factor(["x{}".format(var_idx)],
                                potential=f_x_i)
        ## sampling the subgraph mask first and set cross potentials
        graph_mask = self.subgraph_mask(h_matrix.shape[1])
        
        for var_idx in range(h_matrix.shape[1]):
            
            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                test_condition = np.isclose(np.array([graph_mask[var_idx, var_jdx],
                                                      graph_mask[var_jdx, var_idx]]),
                                            np.array([0,0]))
                
                if not np.all(test_condition):
                    t_potential = - np.array(self.constellation)[None,:].T * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var
                    t_potential = t_potential - t_potential.max()
                    t_ij = np.exp(t_potential)
                    t_ij = t_ij/t_ij.sum()
                    subgraph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                    potential= np.power(t_ij, self.learning_rate))
        return subgraph
        

    
    def fit(self, channel, noise_var, noised_signal, stop_iter=10):
        rate_list = np.linspace(1, 0.01, stop_iter)
        for iti in range(stop_iter):
            # initialize a new graph
            self.learning_rate = rate_list[iti]
            """ set potentials and run message passing"""
            self.graph = self.new_graph(h_matrix=channel,
                                        observation=noised_signal,
                                        noise_var=noise_var)

            # run BP
            iters, converged = self.graph.lbp(normalize=True,
                                              max_iters=50)

            self.first_iter_flag = False


    

class PPBP(LoopyBP):

    def set_potential(self, h_matrix, observation, noise_var):
        power_ratio = noise_var/self.hparam.signal_var
        s = np.matmul(h_matrix.T, h_matrix)
        inv = np.linalg.inv(power_ratio * np.eye(h_matrix.shape[1]) 
                            + s )
        prior_u = inv.dot(h_matrix.T).dot(observation)
                        
        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_x_i = np.exp( (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2)
                             + h_matrix[:, var_idx].dot(observation) * np.array(self.constellation))/noise_var)
            prior_i = np.exp(-0.5 * np.power(self.constellation - prior_u[var_idx], 2) \
                             / (inv[var_idx, var_idx] * noise_var) )
            self.graph.factor(["x{}".format(var_idx)],
                              potential=f_x_i * prior_i )


        for var_idx in range(h_matrix.shape[1]):

            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                t_ij = np.exp(- np.array(self.constellation)[None,:].T
                              * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var)
                self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                  potential=t_ij)
    

class LoopyMP(LoopyBP):
    def __init__(self, noise_var, hparam):
        # get the constellation
        self.constellation = hparam.constellation

        self.n_symbol = hparam.num_tx * 2
        # set the graph
        self.graph = maxsum.mpGraph()
        # add the discrete random variables to graph
        for idx in range(hparam.num_tx * 2):
            self.graph.rv("x{}".format(idx), len(self.constellation))

    
    def set_potential(self, h_matrix, observation, noise_var):
        s = np.matmul(h_matrix.T, h_matrix)
        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_potential = (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2) \
                            + h_matrix[:, var_idx].dot(observation) \
                            * np.array(self.constellation))/noise_var
            f_potential = f_potential - f_potential.max()
            f_x_i = np.exp(f_potential)
            self.graph.factor(["x{}".format(var_idx)],
                              potential=f_x_i)


        for var_idx in range(h_matrix.shape[1]):

            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                t_potential = - np.array(self.constellation)[None,:].T * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var 
                t_potential = t_potential - t_potential.max()
                t_ij = np.exp(t_potential)
                self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                  potential=t_ij)
    

class VariationalBP(LoopyBP):
    def __init__(self, noise_var, hparam):
        self.hparam = hparam
        # get the constellation
        self.constellation = hparam.constellation

        self.n_symbol = hparam.num_tx * 2
        # set the graph
        self.graph = variationalBP.variationalGraph()
        # add the discrete random variables to graph
        for idx in range(hparam.num_tx * 2):
            self.graph.rv("x{}".format(idx), len(self.constellation))


class MMSEalphaBP(AlphaBP):
    def set_potential(self, h_matrix, observation, noise_var):
        power_ratio = noise_var/self.hparam.signal_var
        s = np.matmul(h_matrix.T, h_matrix)
        inv = np.linalg.inv(power_ratio * np.eye(h_matrix.shape[1]) 
                            + s )
        prior_u = inv.dot(h_matrix.T).dot(observation)
                        
        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_potential = (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2)
                             + h_matrix[:, var_idx].dot(observation) * np.array(self.constellation))/noise_var
            f_potential = f_potential - f_potential.max()
            f_x_i = np.exp( f_potential )
            
            p_potential = -0.5 * np.power(self.constellation - prior_u[var_idx], 2) \
                             / (inv[var_idx, var_idx] * noise_var) 
            p_potential = p_potential - p_potential.max()
            prior_i = np.exp(p_potential)
            self.graph.factor(["x{}".format(var_idx)],
                              potential=f_x_i * prior_i)


        for var_idx in range(h_matrix.shape[1]):

            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                t_potential = - np.array(self.constellation)[None,:].T * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var
                t_potential = t_potential - t_potential.max()
                t_ij = np.exp(t_potential)
                self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                  potential=t_ij)

class EPalphaBP(AlphaBP):
    

    def __init__(self, noise_var, hparam):
        super(EPalphaBP, self).__init__(noise_var, hparam)
        # set EP as prior
        self.prior = EP(noise_var, hparam)
        
    def set_potential(self, h_matrix, observation, noise_var, prior_u, prior_var):
        power_ratio = noise_var/self.hparam.signal_var
        s = np.matmul(h_matrix.T, h_matrix)
        
                        
        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_potential = (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2)
                             + h_matrix[:, var_idx].dot(observation) * np.array(self.constellation))/noise_var
            # in case numerial overflow
            f_potential = f_potential - f_potential.max()
            f_x_i = np.exp( f_potential )

            p_potential = -0.5 * np.power(self.constellation - prior_u[var_idx], 2) \
                             / (prior_var[var_idx, var_idx])
            p_potential = p_potential - p_potential.max()

            prior_i = np.exp(p_potential)
            self.graph.factor(["x{}".format(var_idx)],
                              potential=f_x_i * prior_i)


        for var_idx in range(h_matrix.shape[1]):

            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                t_potential = - np.array(self.constellation)[None,:].T * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var
                t_potential = t_potential - t_potential.max()
                t_ij = np.exp(t_potential)
                self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                  potential=t_ij)

    def fit(self, channel, noise_var, noised_signal, stop_iter=10):
        """ set potentials and run message passing"""

        # get prior information first
        self.prior.fit(channel=channel,
                       noise_var=noise_var,
                       noised_signal=noised_signal,
                       stop_iter=10)
               
        self.set_potential(h_matrix=channel,
                           observation=noised_signal,
                           noise_var=noise_var,
                           prior_u=self.prior.mu,
                           prior_var=self.prior.covariance)

        # run BP
        iters, converged = self.graph.lbp(normalize=True)
    

    
class MMSEvarBP(VariationalBP):
    def set_potential(self, h_matrix, observation, noise_var):
        power_ratio = noise_var/self.hparam.signal_var
        s = np.matmul(h_matrix.T, h_matrix)
        inv = np.linalg.inv(power_ratio * np.eye(h_matrix.shape[1]) 
                            + s )
        prior_u = inv.dot(h_matrix.T).dot(observation)
                        
        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_x_i = np.exp( (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2)
                             + h_matrix[:, var_idx].dot(observation) * np.array(self.constellation))/noise_var)
            prior_i = np.exp(-0.5 * np.power(self.constellation - prior_u[var_idx], 2) \
                             / (inv[var_idx, var_idx] * noise_var))
            self.graph.factor(["x{}".format(var_idx)],
                              potential=f_x_i * prior_i)


        for var_idx in range(h_matrix.shape[1]):

            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                t_ij = np.exp(- np.array(self.constellation)[None,:].T
                              * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var)
                self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                  potential=t_ij)
    

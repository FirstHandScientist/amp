'''
Graph implementation and message passing/computation for max-product algorithm.
'''

import code  # code.interact(local=dict(globals(), **locals()))
import logging
import signal
from factorgraph import Graph
from factorgraph import RV 
from factorgraph import Factor as bpFactor
from factorgraph import divide_safezero
# 3rd party
import numpy as np



logger = logging.getLogger(__name__)

# Settings

# Use this to turn all debugging on or off. Intended use: keep on when you're
# trying stuff out. Once you know stuff works, turn off for speed. Can also
# specify when creating each instance, but this global switch is provided for
# convenience.
DEBUG_DEFAULT = True

# This is the maximum number of iterations that we let loopy belief propagation
# run before cutting it off.
LBP_MAX_ITERS = 50

# Otherwise we'd just make some kinda class to do this anyway.
E_STOP = False


##### classes

class mpGraph(Graph):
    def __init__(self, debug=DEBUG_DEFAULT):
        super(mpGraph, self).__init__()
        # add now
        self.debug = debug
        self._rvs = {}

        self._factors = []

    # def rv(self, name, n_opts, labels=[], meta={}, debug=DEBUG_DEFAULT):
    #     rv = RV(name, n_opts, labels, meta, debug)
    #     self.add_rv(rv)
    #     return rv


    def factor(self, rvs, name='', potential=None, meta={},
               debug=DEBUG_DEFAULT):
        # Look up RVs if needed.
        for i in range(len(rvs)):
            if debug:
                assert type(rvs[i]) in [str, unicode, RV]
            if type(rvs[i]) in [str, unicode]:
                rvs[i] = self._rvs[rvs[i]]
            # This is just a coding sanity check.
            assert type(rvs[i]) is RV

        f = Factor(rvs, name, potential, meta, debug)
        self.add_factor(f)
        return f



class Factor(bpFactor):

    def recompute_outgoing(self, normalize=False):
        """message need to be computed in max sum version"""
        # Good old safety.
        if self.debug:
            assert self._outgoing is not None, 'must call init_lbp() first'

        # Save old for convergence check.
        old_outgoing = self._outgoing[:]

        # (Product:) Multiply RV messages into "belief".
        incoming = []
        belief = self._potential.copy()
        for i, rv in enumerate(self._rvs):
            m = rv.get_outgoing_for(self)
            if self.debug:
                assert m.shape == (rv.n_opts,)
            # Reshape into the correct axis (for combining). For example, if
            # our incoming message (And thus rv.n_opts) has length 3, our
            # belief has 5 dimensions, and this is the 2nd (of 5) dimension(s),
            # then we want the shape of our message to be (1, 3, 1, 1, 1),
            # which means we'll use [1, -1, 1, 1, 1] to project our (3,1) array
            # into the correct dimension.
            #
            # Thanks to stackoverflow:
            # https://stackoverflow.com/questions/30031828/multiply-numpy-
            #     ndarray-with-1d-array-along-a-given-axis
            proj = np.ones(len(belief.shape), int)
            proj[i] = -1
            m_proj = m.reshape(proj)
            incoming += [m_proj]
            # Combine to save as we go
            belief *= m_proj

        # Divide out individual belief and (Sum:) add for marginal.
        convg = True
        all_idx = range(len(belief.shape))
        for i, rv in enumerate(self._rvs):
            
            rv_belief = divide_safezero(belief, incoming[i])
            axes = tuple(all_idx[:i] + all_idx[i+1:])
            o = rv_belief.max(axis=axes)
            if self.debug:
                assert self._outgoing[i].shape == (rv.n_opts, )
            if normalize:
                o = divide_safezero(o, sum(o))
            self._outgoing[i] = o
            convg = convg and \
                sum(np.isclose(old_outgoing[i], self._outgoing[i])) == \
                rv.n_opts

        return convg

        

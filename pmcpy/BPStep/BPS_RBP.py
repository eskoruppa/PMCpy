import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from ..chain import Chain
from ..SO3 import so3
from .RBPStiff import GenStiffness
from .BPStep import BPStep

class RBP(BPStep):
    
    def __init__(
        self, 
        chain: Chain, 
        sequence: str, 
        specs: Dict,
        closed: bool = False,
        static_group: bool = True,
        ):
        super().__init__(chain,sequence,specs,closed,static_group)


    def init_params(self) -> None:
        genstiff = GenStiffness(method = self.specs['method'])        
        self.stiffmats = np.zeros((self.nbps,6,6))
        self.gs_vecs   = np.zeros((self.nbps,6))
        if self.closed:
            extseq = self.sequence + self.sequence[0]
        else:
            extseq = self.sequence
        for i in range(self.nbps):
            self.gs_vecs[i]   = genstiff.dimers[extseq[i:i+2]][self.specs['gs_key']]
            self.stiffmats[i] = genstiff.dimers[extseq[i:i+2]][self.specs['stiff_key']]
        self.current_energies  = np.zeros(self.nbps)
        self.proposed_energies = np.zeros(self.nbps)
                    
    def eval_delta_E(self) -> float:
        dE = 0
        for id in self.proposed_ids:
            print(id)
            self.proposed_energies[id] = \
                0.5 * self.proposed_deforms[id].T @ self.stiffmats[id] @ self.proposed_deforms[id]  
            dE += self.proposed_energies[id] - self.current_energies[id]
            print(self.proposed_energies[id] - self.current_energies[id])
        return dE
    
    def set_energy(self, accept: bool = True) -> None:
        if accept:
            self.current_energies = np.copy(self.proposed_energies)
        else:
            self.proposed_energies = np.copy(self.current_energies)

        
        
    
    
if __name__ == '__main__':
    
    npb  = 100
    conf = np.zeros((npb,4,4))
    gs = np.array([0,0,0.6,0,0,0.34])
    g = so3.se3_euler2rotmat(gs)
    conf[0] = np.eye(4)
    for i in range(1,npb):
        conf[i] = conf[i-1] @ g
    
    seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(npb)])
    specs = {'method':'MD','gs_key': 'group_gs','stiff_key': 'group_stiff'}
    
    ch = Chain(conf,keep_backup=True)
    bps = RBP(ch,seq,specs,closed=False,static_group=True)
    
    
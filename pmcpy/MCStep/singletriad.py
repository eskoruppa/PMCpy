import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from abc import ABC, abstractmethod
from ..chain import Chain
from ..BPStep.BPStep import BPStep
from ..ExVol.ExVol import ExVol
from ..SO3 import so3
from .mcstep import MCStep
from ..pyConDec.pycondec import cond_jit
from ..aux import random_unitsphere

class SingleTriad(MCStep):
    
    def __init__(
        self, 
        chain: Chain, 
        bpstep: BPStep, 
        selected_triad_ids: List = None,
        full_trial_conf: bool = False,
        exvol: ExVol = None
        ):
        """
            Initiate:
                - configuration
                - energy model (separate object passed to chain)
        
        """
        super().__init__(
            chain,
            bpstep,
            full_trial_conf,
            exvol=exvol
        )
        self.name = 'SingleTriad'
        
        if selected_triad_ids is None:
            self.selected_triad_ids = np.arange(0,self.nbp)
        else:
            self.selected_triad_ids = np.array(np.sort(selected_triad_ids))
        
        MCS_MST_MAX_THETA = 0.09
        MCS_MST_MAX_TRANS = 0.1
        self.max_theta = MCS_MST_MAX_THETA*np.sqrt(self.chain.temp/300)
        self.max_trans = MCS_MST_MAX_TRANS*np.sqrt(self.chain.temp/300)
        self.closed = self.closed
                
        self.requires_ev_check = True
        self.moved_intervals = np.zeros((1,3))
        self.moved_intervals[0,2] = 1000
        
         
    def mc_move(self) -> bool:
        
        #############################
        # select midstep triads
        id   = self.selected_triad_ids[np.random.randint(len(self.selected_triad_ids))]
        idm1 = (id-1)%self.nbp
        idp1 = (id+1)%self.nbp

        #random translation 
        s = random_unitsphere()
        dr = np.random.uniform(0,self.max_trans) 
        dv = s*dr
        
        #random rotation
        s = random_unitsphere()
        theta = np.random.uniform(0,self.max_theta)
        Theta = s*theta
        G = so3.euler2rotmat(Theta)
        
        tau = np.copy(self.chain.conf[id])
        tau[:3,:3] = G @ tau[:3,:3]
        tau[:3,3] += dv
        
        # propose moves
        if id > 0 or self.closed:
            self.bpstep.propose_move(idm1,self.chain.conf[idm1],tau)
        if id < self.nbps or self.closed:
            self.bpstep.propose_move(id,tau,self.chain.conf[idp1],)
        
        # calculate energy
        dE = self.bpstep.eval_delta_E()
        # print(dE)
        
        # metropolis step
        if np.random.uniform() >= np.exp(-dE):
            return False
    
        ##########################
        # assign changes
        self.chain.conf[id] = tau
        
        self.moved_intervals[0,0] = id
        self.moved_intervals[0,1] = idp1
                        
        return True
 
        
    
    

    
    
    
    
    
if __name__ == '__main__':

    from ..BPStep.BPS_RBP import RBP

    np.set_printoptions(linewidth=250,precision=3,suppress=True)

    npb  = 50
    closed = False
    conf = np.zeros((npb,4,4))
    gs = np.array([0,0,0.6,0,0,0.34])
    g = so3.se3_euler2rotmat(gs)
    conf[0] = np.eye(4)
    for i in range(1,npb):
        g = so3.se3_euler2rotmat(gs+np.random.normal(0,0.1,6))
        conf[i] = conf[i-1] @ g
    
    seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(npb)])
    specs = {'method':'MD','gs_key': 'group_gs','stiff_key': 'group_stiff'}
    
    ch = Chain(conf,keep_backup=True,closed=closed)
    bps = RBP(ch,seq,specs,closed=closed,static_group=True)    


    # tau1 = np.copy(ch.conf[9])
    # tau2 = np.copy(ch.conf[10])
    # tau3 = np.copy(ch.conf[18])
    # tau4 = np.copy(ch.conf[19])
    # tau5 = np.copy(ch.conf[20])

    # cs = Crankshaft(ch,bps,2,16,range_id1=18,range_id2=4)
    ct = SingleTriad(ch,bps)
    Es = []
    
    for i in range(100000):
        ct.mc()
        if i%1000==0:
            print(f'{i}: {ct.acceptance_rate()}')
            Es.append(ct.bpstep.get_total_energy())
            
    
    print(f'<E> / DoFs = {np.mean(Es)/(len(ct.chain.conf)-1)/6}')
    print(np.mean(Es)) 
        
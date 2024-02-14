import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from abc import ABC, abstractmethod
from ..chain import Chain
from ..SO3 import so3


class BPStep(ABC):
    
    def __init__(
        self, 
        chain: Chain, 
        sequence: str, 
        specs: Dict,
        closed: bool = False,
        static_group: bool = True
        ):
        
        self.chain = chain
        self.sequence = sequence
        self.specs = specs
        self.closed = closed
        self.static_group = static_group
        
        self.nbp = len(chain.conf)
        if closed:
            self.nbps = self.nbp
        else:
            self.nbps = self.nbp-1
        
        self.current_deforms  = np.zeros((self.nbps,6))
        self.proposed_deforms = np.zeros((self.nbps,6))
        self.proposed_ids = []
        
        self.init_params()
        self.init_static()
        self.init_conf()
        
    
    def init_conf(self) -> None:
        for id in range(self.nbps):
            self.propose_move(id,self.chain.conf[id],self.chain.conf[(id+1)%self.nbp])
        self.eval_delta_E()
        self.set_move(True)
     
            
    def propose_move(self, id: int, triad1: np.ndarray, triad2: np.ndarray) -> None:
        if self.static_group:
            X = so3.se3_rotmat2euler(self.gs_mats_inv[id] @ so3.se3_triads2rotmat(triad1,triad2))
        else:
            X = so3.se3_rotmat2euler(so3.se3_triads2rotmat(triad1,triad2)) - self.gs_vecs[id]
        
        # print('-----------------')
        # print(np.sum(self.current_deforms[id]-X))
        # print('-----------------')
        
        # print('-----------------')
        # print(self.current_deforms[id])
        # print(X)
        # print('-----------------')
        self.proposed_deforms[id] = X
        self.proposed_ids.append(id)


    def set_move(self, accept: bool = True) -> None:
        if accept:
            self.current_deforms = np.copy(self.proposed_deforms)
        else:
            self.proposed_deforms = np.copy(self.current_deforms)
        self.set_energy(accept)
        self.proposed_ids = []
    
    
    def init_static(self) -> None:
        if self.static_group:
            self.gs_mats     = np.zeros((len(self.gs_vecs),4,4))
            self.gs_mats_inv = np.zeros((len(self.gs_vecs),4,4))
            for i,gs in enumerate(self.gs_vecs):
                self.gs_mats[i]     = so3.se3_euler2rotmat(gs)
                self.gs_mats_inv[i] = so3.se3_inverse(self.gs_mats[i])
    
           
    @abstractmethod
    def init_params(self) -> None:
        """Initialte all parameters necessary for energy evaluation.
            The following variables also need to be set here:
                self.gs_vecs (groundstate vectors) 
        """
        pass
    
    @abstractmethod
    def eval_delta_E(self) -> float:
        pass
    
    @abstractmethod
    def set_energy(self, accept: bool = True) -> None:
        pass
    
    @abstractmethod
    def get_total_energy(self) -> float:
        pass

    

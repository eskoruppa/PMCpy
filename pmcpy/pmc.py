import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from .chain import Chain
from .BPStep.BPStep import BPStep
from .ExVol.ExVol import ExVol
from .MCStep.singletriad import SingleTriad
from .MCStep.crankshaft import Crankshaft
from .MCStep.midstepmove import MidstepMove
from .MCStep.clustertranslation import ClusterTrans
from .SO3 import so3


        
    


if __name__ == '__main__':

    from .BPStep.BPS_RBP import RBP

    np.set_printoptions(linewidth=250,precision=3,suppress=True)

    npb  = 1000
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

    moves = list()

    moves.append(Crankshaft(ch,bps,2,25))
    moves.append(SingleTriad(ch,bps))
    moves.append(ClusterTrans(ch,bps,2,25))
    
    Es = []
    
    confs = []
    confs.append(np.copy(ch.conf[:,:3,3]))
    
    for i in range(1000000):
        for move in moves:
            move.mc()
        
        if i%10000==0:
            print(f'step {i}: ')
            for move in moves:
                print(f'{move.name}: {move.acceptance_rate()}')
            Es.append(bps.get_total_energy())
            confs.append(np.copy(ch.conf[:,:3,3]))
    
    
    print(f'<E> / DoFs = {np.mean(Es)/(len(ch.conf)-1)/6}')   
    
    from .Dumps.xyz import write_xyz
    types = ['C' for i in range(ch.nbp)]
    data = {'pos':confs, 'types':types}
    write_xyz('conf.xyz',data)
    
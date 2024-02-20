import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from ..chain import Chain, se3_triads
from .model_selection import init_bps


from ..BPStep.BPStep import BPStep
from ..ExVol.ExVol import ExVol


from ..MCStep.singletriad import SingleTriad
from ..MCStep.crankshaft import Crankshaft
from ..MCStep.midstepmove import MidstepMove
from ..MCStep.pivot import Pivot
from ..MCStep.clustertranslation import ClusterTrans
from ..SO3 import so3


def equilibrate(
    triads: np.ndarray, 
    positions: np.ndarray, 
    sequence: str,
    closed: bool = False,
    endpoints_fixed: bool = True,
    fixed: List[int] = [], 
    temp: float = 300, 
    num_cycles: int = None,
    # fixed_positions: List[int] = None,
    exvol_rad: float = 0,
    model: str = 'lankas'
    ) -> Dict:
    """_summary_

    Args:
        triads (np.ndarray): _description_
        positions (np.ndarray): _description_
        sequence (str): _description_
        closed (bool, optional): _description_. Defaults to False.
        endpoints_fixed (bool, optional): _description_. Defaults to True.
        fixed (List[int], optional): _description_. Defaults to [].
        temp (float, optional): _description_. Defaults to 300.
        num_cycles (int, optional): _description_. Defaults to None.
        exvol_rad (float, optional): _description_. Defaults to 0.
        model (str, optional): _description_. Defaults to 'lankas'.
    
    TODO:
        - assess convergence of energy
    """
    
    
    exvol_active = False
    keep_backup = False
    exvol = None
    if exvol_rad > 0:
        exvol_active = True
        keep_backup  = True
        print('Warning: Excluded volume interactions not yet implemented!')
        
    
    #############################
    # init configuration chain and energy
    conf  = se3_triads(triads,positions)
    chain = Chain(conf,closed=closed,keep_backup=keep_backup)
    bps   = init_bps(model,chain,sequence,closed=closed,temp=temp)
    N = len(conf)
    
    
    
    #############################
    # init moves
    moves = list()
    
    if endpoints_fixed:
        if 0 not in fixed:
            fixed = [0] + fixed
        if N-1 not in fixed:
            fixed += [N-1]
    
    bpids = [i for i in range(N)]   
    free = [i for i in bpids if i not in fixed]
        
    # add single triad moves:
    single = SingleTriad(chain,bps,selected_triad_ids=free,exvol=exvol)
    Nsingle = len(free)//4
    if Nsingle == 0 and len(free) > 0: Nsingle = 1
    moves += [single for i in range(Nsingle)]

    
    if closed:
        if len(fixed) == 0:
            # add crankshaft
            cs = Crankshaft(chain,bps,2,N//2,exvol=exvol)            
            ct = ClusterTrans(chain,bps,N//2,exvol=exvol)
            moves += [cs,ct]
        else:
            # add crankshaft moves on intervals
            #  -> this will be replaced by a single move with multiple interval assignments
            for fid in range(1,len(fixed)):
                f1 = fixed[fid-1]+1
                f2 = fixed[fid]
                diff = f2-f1
                if diff > 4:
                    rge = np.min([N//2,diff])
                    cs = Crankshaft(chain,bps,2,rge,range_id1=f1,range_id2=f2,exvol=exvol)
                    ct = ClusterTrans(chain,bps,2,rge,range_id1=f1,range_id2=f2,exvol=exvol)
                    moves += [cs,ct] 
            
            # between last and first fix
            f1 = (fixed[-1]+1)
            f2 = fixed[0]
            diff = f2 - f1 + N
            if diff > 4:
                rge = np.min([N//2,diff])
                cs = Crankshaft(chain,bps,2,rge,range_id1=f1,range_id2=f2,exvol=exvol)
                ct = ClusterTrans(chain,bps,2,rge,range_id1=f1,range_id2=f2,exvol=exvol)
                moves += [cs,ct] 
               
    else:     
        if not endpoints_fixed:
            # endpoints open
            if len(fixed) == 0:
                pv1 = Pivot(chain,bps,rotate_end=False,exvol=exvol)
                pv2 = Pivot(chain,bps,rotate_end=True,exvol=exvol)
                moves += [pv1,pv2]
            else:
                if fixed[0] > 4:
                    pv1 = Pivot(chain,bps,rotate_end=False,exvol=exvol,selection_limit_id=fixed[0])
                    moves.append(pv1)
                if fixed[-1] < N - 5:
                    pv2 = Pivot(chain,bps,rotate_end=True,exvol=exvol,selection_limit_id=fixed[-1]+1)
                    moves.append(pv2)
          
        else:
            for fid in range(1,len(fixed)):
                f1 = fixed[fid-1]+1
                f2 = fixed[fid]
                diff = f2-f1
                if diff > 4:
                    rge = np.min([N//2,diff])
                    cs = Crankshaft(chain,bps,2,rge,range_id1=f1,range_id2=f2,exvol=exvol)
                    ct = ClusterTrans(chain,bps,2,rge,range_id1=f1,range_id2=f2,exvol=exvol)
                    moves += [cs,ct] 
                    
    #############################
    # simulation loop
    if num_cycles is None:
        num_cycles = N*50
    
    confs = []
    confs.append(np.copy(chain.conf[:,:3,3]))
    
    print(f'{len(moves)} moves initated')
    
    for cyc in range(num_cycles):
        for move in moves:
            move.mc()
        
        if cyc%100==0:
            confs.append(np.copy(chain.conf[:,:3,3]))
        if cyc%1000==0: 
            print(f'cycle {cyc}: ')
            
    out = {
        'positions' : chain.conf[:,:3,3],
        'triads'    : chain.conf[:,:3,:3],
        'elastic'   : bps.get_total_energy(),
        'confs'     : confs
    }
    return out
    
    
                
            

if __name__ == '__main__':

    np.set_printoptions(linewidth=250,precision=3,suppress=True)

    npb  = 100
    closed = False
    endpoints_fixed = True
    fixed = []
    temp = 300
    
    conf = np.zeros((npb,4,4))
    gs = np.array([0,0,0.6,0,0,0.34])
    g = so3.se3_euler2rotmat(gs)
    conf[0] = np.eye(4)
    for i in range(1,npb):
        g = so3.se3_euler2rotmat(gs+np.random.normal(0,0.1,6))
        conf[i] = conf[i-1] @ g
    
    seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(npb)])

    triads = conf[:,:3,:3]
    pos    = conf[:,:3,3]

    print('first')
    out = equilibrate(triads,pos,seq,closed=closed,endpoints_fixed=False,fixed=[],temp=100000,num_cycles=100)

    fixed = [10,20,30,70]

    fixed_pos1 = np.copy(out['positions'][fixed])

    print('second')
    out = equilibrate(out['triads'],out['positions'],seq,closed=closed,endpoints_fixed=endpoints_fixed,fixed=fixed,temp=300)

    fixed_pos2 = np.copy(out['positions'][fixed])
    
    print(fixed_pos2-fixed_pos1)

    from ..Dumps.xyz import write_xyz
    types = ['C' for i in range(len(conf))]
    data = {'pos':out['confs'], 'types':types}
    write_xyz('test_equi.xyz',data)
    
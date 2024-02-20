import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict

from ..chain import Chain
from ..BPStep.BPStep import BPStep
from ..BPStep.BPS_RBP import RBP


def init_bps(model: str, chain: Chain, sequence: str, closed: bool = False, temp: float = 300):
    
    if model.lower() in ['lankas','rbp_md']:
        specs = {'method':'MD','gs_key': 'group_gs','stiff_key': 'group_stiff'}
        return RBP(chain,sequence,specs,closed=closed,static_group=True,temp=temp) 

    
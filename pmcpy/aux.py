import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from .pyConDec.pycondec import cond_jit

@cond_jit
def random_unitsphere():
    a = 2*np.pi*np.random.uniform()
    x = np.random.uniform()
    b = np.arccos(1-2*x)
    vec = np.ones(3)
    vec[:2] *= np.sin(b)
    vec[2]  *= np.cos(b)
    vec[0]  *= np.cos(a)
    vec[1]  *= np.sin(a)
    return vec
    

        
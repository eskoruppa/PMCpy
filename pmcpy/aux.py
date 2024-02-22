import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .pyConDec.pycondec import cond_jit
from .SO3 import so3


@cond_jit
def random_unitsphere():
    a = 2 * np.pi * np.random.uniform()
    x = np.random.uniform()
    b = np.arccos(1 - 2 * x)
    vec = np.ones(3)
    vec[:2] *= np.sin(b)
    vec[2] *= np.cos(b)
    vec[0] *= np.cos(a)
    vec[1] *= np.sin(a)
    return vec


def params2conf(params: np.ndarray) -> np.ndarray:
    conf = np.zeros((len(params), 4, 4))
    conf[0] = np.eye(4)
    for i in range(1, len(params)):
        conf[i] = conf[i - 1] @ so3.se3_euler2rotmat(params[i])
    return conf

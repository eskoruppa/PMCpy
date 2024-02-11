import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from abc import ABC, abstractmethod
from ..chain import Chain


class ExVol(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def check(self, moved: List = None):
        pass
    
    
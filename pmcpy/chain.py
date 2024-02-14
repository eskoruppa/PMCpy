import sys, os
import numpy as np
from typing import List, Tuple, Callable, Any, Dict


class Chain:
    
    def __init__(
        self, 
        configuration: np.ndarray,
        closed: bool = False,
        keep_backup: bool = False
        ):
        """
            Initiate:
                - configuration
                - energy model (separate object passed to chain)
        
        """
        
        self.conf = configuration
        self.closed = closed
        self.keep_backup = keep_backup
        self.temp = 300
        
        self.nbp = len(self.conf)
        self.nbps = self.nbp
        if not self.closed:
            self.nbps -= 1


    
    def set_conf(self, conf: np.ndarray) -> None:
        self.conf = conf
        self.set_backup()
        
    def set_backup(self, conf: np.ndarray = None) -> None:
        if self.keep_backup:
            if conf is None:
                self.backup_conf = np.copy(self.conf)
            else:
                self.backup_conf = np.copy(conf) 
     
        
    

        
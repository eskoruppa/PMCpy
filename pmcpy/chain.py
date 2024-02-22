import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .SO3 import so3


class Chain:
    def __init__(
        self, configuration: np.ndarray, closed: bool = False, keep_backup: bool = False
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

    def realign_triads(self):
        for i in range(self.nbp):
            self.conf[i] = so3.se3_euler2rotmat(so3.se3_rotmat2euler(self.conf[i]))


def se3_triads(triads: np.ndarray, positions: np.ndarray):
    if len(positions) != len(triads):
        raise ValueError(
            f"Mismatched dimensions of positions ({positions.shape}) and triads ({triads.shape})."
        )

    conf = np.zeros((len(positions), 4, 4))
    conf[:, :3, :3] = triads
    conf[:, :3, 3] = positions
    conf[:, 3, 3] = 1
    return conf

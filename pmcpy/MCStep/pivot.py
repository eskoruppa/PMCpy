import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..aux import params2conf, random_unitsphere
from ..BPStep.BPStep import BPStep
from ..chain import Chain
from ..ExVol.ExVol import ExVol
from ..pyConDec.pycondec import cond_jit
from ..SO3 import so3
from .mcstep import MCStep


class Pivot(MCStep):
    def __init__(
        self,
        chain: Chain,
        bpstep: BPStep,
        selection_limit_id: int = None,
        rotate_end: bool = True,
        full_trial_conf: bool = False,
        exvol: ExVol = None,
        numba_tail_rotation: bool = True,
    ):
        """
        Initiate:
            - configuration
            - energy model (separate object passed to chain)

        """
        super().__init__(chain, bpstep, full_trial_conf, exvol=exvol)
        self.name = "Pivot"
        self.numba_tail_rotation = numba_tail_rotation

        if self.closed:
            raise ValueError(f"Pivot: Pivot move not allowed for closed chains.")

        MCS_MST_MAX_THETA = 0.09
        self.max_theta = MCS_MST_MAX_THETA * np.sqrt(self.chain.temp / 300)

        self.limited_range = False
        self.rotate_end = rotate_end
        if selection_limit_id is not None:
            self.limited_range = True
            self.limit_id = selection_limit_id

        self.requires_ev_check = True

        self.moved_intervals = np.zeros((1, 3))
        if self.rotate_end:
            self.moved_intervals[0, 1] = self.nbps
        self.moved_intervals[0, 2] = 1000

    #########################################################################################
    #########################################################################################

    def mc_move(self) -> bool:
        ##########################################################
        ##########################################################
        # rotate back
        if self.rotate_end:
            #############################
            # select hinges
            if self.limited_range:
                id = np.random.randint(self.limit_id, self.nbp)
            else:
                id = np.random.randint(0, self.nbp)
                # id = np.random.randint(1,self.nbp)
            idm1 = id - 1

            # generate random rotation
            s = random_unitsphere()
            theta = np.random.uniform(0, self.max_theta)
            Theta = s * theta
            G = so3.euler2rotmat(Theta)

            # rotate triad
            tau = np.copy(self.chain.conf[id])
            tau[:3, :3] = G @ tau[:3, :3]

            if id > 0:
                # propose moves
                self.bpstep.propose_move(idm1, self.chain.conf[idm1], tau)

            # calculate energy
            dE = self.bpstep.eval_delta_E()

            # metropolis step
            if np.random.uniform() >= np.exp(-dE):
                return False

            self.chain.conf[id] = tau
            if id == self.nbps:
                self.moved_intervals[0, 0] = id
                return True

            if self.numba_tail_rotation:
                # numba implementation of tail rotation
                back_tail_rotation(self.chain.conf, G, self.nbp, id)
            else:
                # python implementation
                self.chain.conf[id + 1 :, :3, :3] = (
                    G @ self.chain.conf[id + 1 :, :3, :3]
                )
                vecs = self.chain.conf[id + 1 :, :3, 3] - self.chain.conf[id:-1, :3, 3]
                rotated_vecs = (G @ vecs.T).T
                for i in range(0, self.nbp - id - 1):
                    self.chain.conf[id + 1 + i, :3, 3] = (
                        self.chain.conf[id + i, :3, 3] + rotated_vecs[i]
                    )

            self.moved_intervals[0, 0] = id
            return True

        ##########################################################
        ##########################################################
        # rotate front
        else:
            #############################
            # select hinges
            if self.limited_range:
                id = np.random.randint(0, self.limit_id)
            else:
                id = np.random.randint(0, self.nbp)
                # id = np.random.randint(1,self.nbp)
            idp1 = id + 1

            # generate random rotation
            s = random_unitsphere()
            theta = np.random.uniform(0, self.max_theta)
            Theta = s * theta
            G = so3.euler2rotmat(Theta)

            # rotate triad
            tau = np.copy(self.chain.conf[id])
            tau[:3, :3] = G @ tau[:3, :3]

            if id < self.nbps:
                # propose moves
                self.bpstep.propose_move(id, tau, self.chain.conf[idp1])

            # calculate energy
            dE = self.bpstep.eval_delta_E()

            # metropolis step
            if np.random.uniform() >= np.exp(-dE):
                return False

            self.chain.conf[id] = tau
            if id == 0:
                self.moved_intervals[0, 1] = 0
                return True

            if self.numba_tail_rotation:
                # numba implementation of tail rotation
                front_tail_rotation(self.chain.conf, G, id)
            else:
                # python implementation
                self.chain.conf[:id, :3, :3] = G @ self.chain.conf[:id, :3, :3]
                vecs = self.chain.conf[1:idp1, :3, 3] - self.chain.conf[:id, :3, 3]
                rotated_vecs = (G @ vecs.T).T
                for i in range(0, id):
                    self.chain.conf[id - 1 - i, :3, 3] = (
                        self.chain.conf[id - i, :3, 3] - rotated_vecs[id - 1 - i]
                    )

            self.moved_intervals[0, 1] = id
            return True


@cond_jit
def back_tail_rotation(
    conf: np.ndarray,
    G: np.ndarray,
    nbp: int,
    id: int,
) -> np.ndarray:
    vec = np.zeros((nbp - id - 1, 3))
    for i in range(nbp - id - 1):
        vec[i] = conf[id + 1 + i, :3, 3] - conf[id + i, :3, 3]
    for i in range(nbp - id - 1):
        conf[id + 1 + i, :3, :3] = G @ conf[id + 1 + i, :3, :3]
        conf[id + 1 + i, :3, 3] = conf[id + i, :3, 3] + G @ vec[i]
    return conf


@cond_jit
def front_tail_rotation(
    conf: np.ndarray,
    G: np.ndarray,
    id: int,
) -> np.ndarray:
    vec = np.zeros((id, 3))
    for i in range(id):
        vec[i] = G @ (conf[i + 1, :3, 3] - conf[i, :3, 3])
    for i in range(id):
        conf[id - 1 - i, :3, :3] = G @ conf[id - 1 - i, :3, :3]
        conf[id - 1 - i, :3, 3] = conf[id - i, :3, 3] - vec[id - 1 - i]
    return conf


if __name__ == "__main__":
    from ..BPStep.BPS_RBP import RBP

    np.set_printoptions(linewidth=250, precision=3, suppress=True)

    nbp = 100
    closed = False
    conf = np.zeros((nbp, 4, 4))
    gs = [np.array([0, 0, 0.6, 0, 0, 0.34]) for i in range(nbp - 1)]

    # g = so3.se3_euler2rotmat(gs)
    # conf[0] = np.eye(4)
    # for i in range(1,nbp):
    #     # g = so3.se3_euler2rotmat(gs+np.random.normal(0,0.1,6))
    #     g = so3.se3_euler2rotmat(gs)
    #     conf[i] = conf[i-1] @ g

    conf = params2conf(gs)

    seq = "".join(["ATCG"[np.random.randint(4)] for i in range(nbp)])
    specs = {"method": "MD", "gs_key": "group_gs", "stiff_key": "group_stiff"}

    ch = Chain(conf, keep_backup=True, closed=closed)
    bps = RBP(ch, seq, specs, closed=closed, static_group=True)

    selection_limit_id = 50
    rotate_end = False

    # cs = Crankshaft(ch,bps,2,16,range_id1=18,range_id2=4)
    cs = Pivot(ch, bps, selection_limit_id=selection_limit_id, rotate_end=rotate_end)
    moves = []
    moves.append(cs)

    Es = []
    confs = []
    confs.append(np.copy(ch.conf[:, :3, 3]))

    import time

    for move in moves:
        move.mc()

    t1 = time.time()

    for i in range(10000):
        for move in moves:
            move.mc()

        if i % 100 == 0:
            print(f"step {i}: ")
            for move in moves:
                print(f"{move.name}: {move.acceptance_rate()}")
            Es.append(bps.get_total_energy())
            confs.append(np.copy(ch.conf[:, :3, 3]))
            if not bps.check_deform_consistency():
                print("inconsistent")
                sys.exit()

    t2 = time.time()
    print(f"dt = {t2-t1}")

    print(f"<E> / DoFs = {np.mean(Es)/(len(cs.chain.conf)-1)/6}")

    from ..Dumps.xyz import write_xyz

    types = ["C" for i in range(ch.nbp)]
    data = {"pos": confs, "types": types}
    write_xyz("pivot.xyz", data)

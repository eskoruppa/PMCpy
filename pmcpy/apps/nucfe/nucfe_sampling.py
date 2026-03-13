import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ...utils import params2conf
# from ...BPStep.BPS_RBP import RBP
from ...BPStep.BPS_LRBP import LRBP
from ...BPStep.BPStep import BPStep
from ...chain import Chain
from ...Dumps.xyz import write_xyz
from ...ExVol.ExVol import ExVol
from ...MCStep.clustertranslation import ClusterTrans
from ...MCStep.crankshaft import Crankshaft
from ...MCStep.midstepmove import MidstepMove
from ...MCStep.pivot import Pivot
from ...MCStep.singletriad import SingleTriad
from ...SO3 import so3
from .read_nuc_conf import read_nucleosome_triads


def calculate_midstep_triads(
    triad_ids: List[
        int
    ],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray,
) -> np.ndarray:
    midstep_triads = np.zeros((len(triad_ids), 4, 4))
    for i, id in enumerate(triad_ids):
        T1 = nucleosome_triads[id]
        T2 = nucleosome_triads[id + 1]
        midstep_triads[i, :3, :3] = T1[:3, :3] @ so3.euler2rotmat(
            0.5 * so3.rotmat2euler(T1[:3, :3].T @ T2[:3, :3])
        )
        midstep_triads[i, :3, 3] = 0.5 * (T1[:3, 3] + T2[:3, 3])
        midstep_triads[i, 3, 3] = 1
    return midstep_triads


def mutation_sampling(moves, bps_ensemble, curid, total_ids, steps):
    total_ids = len(bps_ensemble)
    for m in range(steps):
        for move in moves:
            move.mc()

    Ecurr = bps_ensemble[curid].get_total_energy()

    # new_id = np.random.randint(total_ids)
    change = np.random.randint(0,2)*2-1
    new_id = (curid + change) % total_ids

    bps_ensemble[new_id].chain = bps_ensemble[curid].chain
    bps_ensemble[new_id].init_conf()
    Enew = bps_ensemble[new_id].get_total_energy()

    dE = Enew - Ecurr
    if np.random.uniform() < np.exp(-dE):
        bps = bps_ensemble[new_id]
        for move in moves:
            move.bpstep = bps
        # print(f'Moved {curid} -> {new_id}')
        curid = new_id
        bps_ensemble[curid].init_conf()
    return curid


if __name__ == "__main__":
    
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    
    outfn = os.path.join(os.path.dirname(__file__), "out")
    seqfn = os.path.join(os.path.dirname(__file__), "seqs")
    triadfn = os.path.join(os.path.dirname(__file__), "State/Nucleosome.state")

    num_cycls_per_step = 2
    steps = 500000
    
    conf = read_nucleosome_triads(triadfn)
    refconf = np.copy(conf)
    midstep_ids = [
        2, 6, 14, 17, 24, 29, 34, 38, 45, 49, 55,
        59, 65, 69, 76, 80, 86, 90, 96, 100, 107,
        111, 116, 121, 128, 131, 139, 143
    ]
    
    misteps = calculate_midstep_triads(midstep_ids, conf)

    closed = False
    specs = {"method": "MD", "gs_key": "group_gs", "stiff_key": "group_stiff"}
    chain = Chain(conf, keep_backup=True, closed=closed)

    ##############################################################
    ##############################################################
    ##############################################################
    # load ensemble

    bps_ensemble = []
    with open(seqfn, "r") as f:
        line = f.readline()
        while line != "":
            seq = line.strip()
            bps_ensemble.append(
                LRBP(chain, seq, specs, closed=closed, static_group=True)
            )
            line = f.readline()
            # print(seq)

    total_ids = len(bps_ensemble)
    curid = np.random.randint(total_ids)
    bps = bps_ensemble[curid]

    ##############################################################
    ##############################################################
    ##############################################################

    moves = []
    # add midstep moves
    msm = MidstepMove(chain, bps, midstep_ids)
    for i in range(14):
        moves.append(msm)

    # add cluster translation moves and crankshaft moves
    for i in range(len(midstep_ids) - 1):
        if midstep_ids[i + 1] - midstep_ids[i] >= 5:
            idfrom = midstep_ids[i] + 2
            idto = midstep_ids[i + 1]
            print(idfrom, idto)
            moves.append(
                ClusterTrans(chain, bps, 2, 8, range_id1=idfrom, range_id2=idto)
            )
            moves.append(Crankshaft(chain, bps, 2, 8, range_id1=idfrom, range_id2=idto))

    # moves = [Crankshaft(chain,bps,2,8,range_id1=19,range_id2=24)]

    # add single moves
    ids = [0, 1]
    for i in range(len(midstep_ids) - 1):
        ids += [i for i in range(midstep_ids[i] + 2, midstep_ids[i + 1])]
    ids += [145, 146]
    stm = SingleTriad(chain, bps, selected_triad_ids=ids)
    for i in range(28):
        moves.append(stm)

    print('Equilibrating')
    for i in range(1000):
        if i % 100 == 0:
            print(f"step {i}")
        curid = mutation_sampling(moves, bps_ensemble, curid, total_ids, num_cycls_per_step)
        
    # Es = []
    # confs = []
    # confs.append(np.copy(chain.conf[:, :3, 3]))

    import time
    t1 = time.time()

    locations = list()

    for i in range(steps):
        curid = mutation_sampling(moves, bps_ensemble, curid, total_ids, num_cycls_per_step)
        locations.append(curid)
        
        # print(np.sum(misteps-calculate_midstep_triads(midstep_ids, chain.conf)))

        # print('###########################')
        # for i in range(len(chain.conf)):
        #     print(i)
        #     print(np.sum(refconf[i]-chain.conf[i]))
        #     if np.abs(np.sum(refconf[i]-chain.conf[i])) < 1e-8:
        #         print('did not change')
        #         sys.exit()

        if i % 10 == 0:
            print(f"step {i}")
            print(f"Current id: {curid}")

        if i % 1000 == 0:
            print(f"step {i}")
            print(f"Current id: {curid}")
            t2 = time.time()
            print(f"dt = {t2-t1}")
            t1 = time.time()

            with open(outfn, "a") as f:
                for loc in locations:
                    f.write(f"{loc}\n")
            locations = []

            chain.realign_triads()

    with open(outfn, "a") as f:
        for loc in locations:
            f.write(f"{loc}\n")

    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(8.6 / 2.54, 5.0 / 2.54))
    # ax1 = fig.add_subplot(111)
    # ax1.hist(locations, density=True, bins=np.arange(len(bps_ensemble)))
    # plt.show()

    # print(f'<E> / DoFs = {np.mean(Es)/(len(conf)-1)/6}')
    # types = ['C' for i in range(chain.nbp)]
    # data = {'pos':confs, 'types':types}
    # write_xyz('nuc.xyz',data)

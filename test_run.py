#!/usr/bin/python
import os,sys
from typing import Dict, List, Tuple, Any, Callable
import numpy as np

import pmcpy

np.set_printoptions(linewidth=250, precision=3, suppress=True)

def gen_circular(nbp: int, lk: float, disc_len: float):
    conf = np.zeros((nbp,4,4))
    conf[:,3,3] = 1
    dtheta = 2*np.pi / nbp
    conf[:,:3,3] = [[np.cos(i*dtheta),np.sin(i*dtheta),0] for i in range(nbp)]
    conf[:,:3,3] *= disc_len / np.linalg.norm(conf[1,:3,3]-conf[0,:3,3])
    for i in range(nbp):
        t = conf[(i+1)%nbp,:3,3] - conf[i,:3,3] 
        t = t / np.linalg.norm(t)
        b = np.array([0,0,1])
        n = np.cross(b,t)
        conf[i,:3,0] = n
        conf[i,:3,1] = b
        conf[i,:3,2] = t
    
    thpbp = lk*2*np.pi / nbp
    for i in range(nbp):
        theta = i*thpbp
        R = pmcpy.so3.euler2rotmat(np.array([0,0,theta]))
        conf[i,:3,:3] = conf[i,:3,:3] @ R 
    # plot(conf[:,:3,3],conf[:,:3,:3]=
    return conf 

endpoints_fixed = False
fixed = []

nbp = 315
dlk = 5
disc_len = 0.34
closed = True
temp = 300
exvol_rad = 2

lk = nbp//10.5 + dlk    
conf = gen_circular(nbp,lk,disc_len)
seq = "".join(["ATCG"[np.random.randint(4)] for i in range(nbp)])

triads = conf[:, :3, :3]
pos = conf[:, :3, 3]

print(f'link = {pmcpy.pylk.triads2link(pos,triads)}')

run = pmcpy.Run(
    triads,
    pos,
    seq,
    closed=closed,
    endpoints_fixed=True,
    fixed=[],
    temp=300,
    exvol_rad=exvol_rad,
    check_crossings=True,
    parameter_set='md'
)

out = run.equilibrate(equilibrate_writhe=True,dump_every=10)

# from ..Dumps.xyz import write_xyz

# print(out["confs"][:,:,:3,3].shape)

# types = ["C" for i in range(len(conf))]
# data = {"pos": out["confs"][:,:,:3,3], "types": types}
# write_xyz("test_equi.xyz", data)



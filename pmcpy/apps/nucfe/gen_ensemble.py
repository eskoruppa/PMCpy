import os
import sys

import numpy as np

from ...SO3 import so3

if __name__ == "__main__":
    np.set_printoptions(linewidth=250, precision=3, suppress=True)

    seq = "".join(["ATCG"[np.random.randint(4)] for i in range(147)])

    num_seqs = 200
    perms_per_seq = 1 
    close_ensemble = True
    
    fn = os.path.join(os.path.dirname(__file__), "seqs5")

    ##############################################################
    ##############################################################
    ##############################################################
    # gen bps
    seqs = []
    seqs.append(seq)
    print(seq)
    for i in range(num_seqs-1):
        
        if perms_per_seq == 1:
        
            id = np.random.randint(147)
            prev = seq[id]
            remaining = "ATCG".replace(prev, "")

            nseq = ""
            if id > 0:
                nseq += seq[:id]
            nseq += remaining[np.random.randint(3)]
            if id < len(seq) - 1:
                nseq += seq[id + 1 :]
            seq = nseq
            ph = "".join(" " for i in range(id))
            print(ph + "^")
            print(seq)
            seqs.append(seq)
            
        else:
            changeids = np.arange(len(seq))
            np.random.shuffle(changeids)
            changeids = changeids[:perms_per_seq]

            seql = [seq[i] for i in range(len(seq))]
            for cid in changeids:
                seql[cid] = "ATCG".replace(seql[cid], "")[np.random.randint(3)]   
            seq = ''.join(seql)
            seqs.append(seq)
        
    
    # print(seqs[0])
    # print(seqs[-1])
    
    if close_ensemble:
        diff_ids = [i for i in range(len(seqs[0])) if seqs[0][i] != seqs[-1][i]]
        
        ndiffs = len(diff_ids)
        for i in range(ndiffs-1):
            last_seq = seqs[-1]
            id = diff_ids[np.random.randint(len(diff_ids))]
            nseq = ""
            if id > 0:
                nseq += last_seq[:id]
            nseq += seqs[0][id]
            if id < len(seqs[0])-1:
                nseq += last_seq[id+1:]
            ph = "".join(" " for i in range(id))
            print(ph + "^")
            print(nseq)    
            
            seqs.append(nseq)
            diff_ids.remove(id)
    
    print(len(seqs))
    with open(fn, "w") as f:
        for seq in seqs:
            f.write(seq + "\n")

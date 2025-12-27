"""
Precompute and save the wigner 3j symbols for computing alm variance.
"""
from importlib import resources
import numpy as np
import healpy as hp
from ducc0.misc import wigner3j_int

from mpi4py import MPI


def compute_l(l, Lmax):
    out = np.zeros((l+1, Lmax+1), dtype=np.float32)
    w3j0 = wigner3j_int(l, l, 0, 0)[1]
    for m in range(0, l + 1):
        l0, w3j1 = wigner3j_int(l, l, m, -m)
        n = np.minimum(w3j1.shape[0], Lmax - l0)
        L = np.arange(l0, l0 + n)
        fac = (2*l + 1)*np.sqrt(2*L + 1)/np.sqrt(4*np.pi)*(-1)**m
        out[m, L] = (fac*w3j1[:n]*w3j0[L]).real
    return out


if __name__ == "__main__":
    """
    mpirun -n 48 python prepare_w3jllmm.py
    """
    lmax = 5000
    Lmax = 50

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        buffer = np.zeros((hp.Alm.getsize(lmax), Lmax+1), dtype=np.float32)
        status = MPI.Status()

        received = 0
        expected = lmax + 1

        while received<expected:
            l = comm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
            src = status.Get_source()
            buf = np.empty((l+1, Lmax+1), dtype=np.float32)
            comm.Recv(buf, source=src, tag=1)
            for m in range(0, l + 1):
                idx = hp.Alm.getidx(lmax, l, m)
                buffer[idx, :] = buf[m]
            received += 1
        fname = str(resources.files('healqest')/'data'/'w3jllmm.npy')
        np.save(fname, buffer)
    else:
        for l in range(rank-1, lmax+1, size-1):
            out = compute_l(l, Lmax)
            # Send location info
            comm.send(l, dest=0, tag=0)
            # Send actual data
            comm.Send(out, dest=0, tag=1)

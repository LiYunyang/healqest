import cProfile
import numpy as np
import healpy as hp
from healqest import weights, resp, wignerd

lmax = 3000
Lmax = 3000
ell = np.arange(lmax + 1)
Cls = np.array([2*(1 + (ell/2000)**-0.2),
                0*ell,
                1e-2*(1 + (ell/1000)**-1),
                1e-2*(1 + (ell/1000)**-1),
                ])
Cls[:, :2] = 0

np.random.seed(37)
almbar = {s: _ for s, _ in zip('TEB', hp.synalm(Cls, lmax=lmax, ))}
cls = dict(tt=Cls[0], te=Cls[1], ee=Cls[2], bb=Cls[3])

qe1, qe2 = 'TT', 'TT'
qeXY = weights.weights_plus(qe1, cls, lmax)
qeZA = weights.weights_plus(qe2, cls, lmax)
XZ = hp.alm2cl(almbar[qe1[0]], almbar[qe2[0]])
YA = hp.alm2cl(almbar[qe1[1]], almbar[qe2[1]])
XA = hp.alm2cl(almbar[qe1[0]], almbar[qe2[1]])
YZ = hp.alm2cl(almbar[qe1[1]], almbar[qe2[0]])
ret = np.zeros(Lmax + 1, dtype=np.complex128)


def target():
    return resp.fill_clq1q2_fullsky(qeXY, qeZA, ret, XZ, YA, XA, YZ)


# Profile the function
profiler = cProfile.Profile()
profiler.enable()
N0 = target()  # Call the target function
profiler.disable()
profiler.print_stats(sort="cumulative")


print("expectation:", [12.03002089+0.j, 15.94917429+0.j, 19.86296193+0.j])
print("result", np.log10(N0[[10, 100, 1000]]))

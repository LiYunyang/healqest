import time
import numpy as np


def cli(cl):
    ret = np.zeros_like(cl)
    good = np.logical_and(cl != 0, ~np.isnan(cl))
    np.reciprocal(cl, out=ret, where=good)
    return ret


def invert_teb(teb, te=None):
    """
    Compute the inverse of the TEB covariance where only TE correlations are non-zero.

    Parameters
    ----------
    teb : np.ndarray
        shape (3, ..., lmax+1), for TT/EE/BB
    te: np.ndarray, optional
        shape(..., lmax+1, ), for TE terms.
    """
    if te is None:
        return cli(teb)
    else:
        assert teb.shape[0] == 3
        assert teb.shape[-1] == te.shape[-1]
        bb = cli(teb[2])
        norm = cli(teb[0] * teb[1] - te**2)
        teb_out = np.array([teb[1] * norm, teb[0] * norm, bb])
        te_out = -te * norm
        # special care for TE, as some "norm" might be 0 where the numerators are NaN
        bad = np.logical_or(norm == 0, np.isnan(norm))
        teb_out[:2, bad] = 0
        te_out[bad] = 0
        return teb_out, te_out


class DeltaTime(object):
    """helper class to contain / print a time difference."""

    def __init__(self, _dt: float):
        self.dt = _dt

    def __str__(self):
        total_seconds = int(self.dt)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def __int__(self):
        return int(self.dt)


class StopWatch(object):
    """simple stopwatch timer class."""

    def __init__(self):
        self.st = time.time()
        self.lt = self.st

    def lap(self):
        """Return the time since start and the time since last call to lap or elapsed."""
        lt = time.time()
        ret = (DeltaTime(lt - self.st), DeltaTime(lt - self.lt))
        self.lt = lt
        return ret

    def elapsed(self):
        """Return the time since initialization."""
        lt = time.time()
        ret = DeltaTime(lt - self.st)
        self.lt = lt
        return ret

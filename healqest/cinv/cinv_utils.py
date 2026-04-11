import sys, time, os
import numpy as np
from contextlib import contextmanager



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
        norm = cli(teb[0]*teb[1]-te**2)
        teb_out = np.array([teb[1]*norm, teb[0]*norm, bb])
        te_out = -te*norm
        # special care for TE, as some "norm" might be 0 where the numerators are NaN
        bad = np.logical_or(norm==0, np.isnan(norm))
        teb_out[:2, bad] = 0
        te_out[bad] = 0
        return teb_out, te_out



class DeltaTime(object):
    """helper class to contain / print a time difference."""

    def __init__(self, _dt):
        self.dt = _dt

    def __str__(self):
        return "%02d:%02d:%02d" % (
            np.floor(self.dt / 60 / 60),
            np.floor(np.mod(self.dt, 60 * 60) / 60),
            np.floor(np.mod(self.dt, 60)),
        )

    def __int__(self):
        return int(self.dt)


class StopWatch(object):
    """simple stopwatch timer class."""

    def __init__(self):
        self.st = time.time()
        self.lt = self.st

    def lap(self):
        """return the time since start and the time since last call to lap or elapsed."""

        lt = time.time()
        ret = (DeltaTime(lt - self.st), DeltaTime(lt - self.lt))
        self.lt = lt
        return ret

    def elapsed(self):
        """return the time since initialization."""

        lt = time.time()
        ret = DeltaTime(lt - self.st)
        self.lt = lt
        return ret


# -- below cribbed from
# http://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python


@contextmanager
def stdout_redirected(to=os.devnull):
    try:
        __IPYTHON__
    except NameError:
        pass
    else:
        to = sys.stdout  # do not redirect in python

    if to is not sys.stdout:
        fd = sys.stdout.fileno()

        def _redirect_stdout(nto):
            sys.stdout.close()  # + implicit flush()
            os.dup2(nto.fileno(), fd)  # fd writes to 'to' file
            sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

        with os.fdopen(os.dup(fd), "w") as old_stdout:
            with open(to, "w") as file:
                _redirect_stdout(nto=file)
            try:
                yield  # allow code to be run with the redirected stdout
            finally:
                _redirect_stdout(nto=old_stdout)  # restore stdout.
                # buffering and flags such as
                # CLOEXEC may be different
    else:
        sys.stdout.flush()
        try:
            yield
        finally:
            pass

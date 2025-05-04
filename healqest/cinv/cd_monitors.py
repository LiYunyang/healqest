import numpy as np
from . import cinv_utils
import logging
logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0


class Logger:
    def __init__(self, **kwargs):
        pass

    def __call__(self, i, eps, watch=None, **kwargs):
        pass


class logger_basic(Logger):
    def __call__(self, i, eps, watch=None, **kwargs):
        if rank ==0:
            logger.info(f"[{str(watch.elapsed())}] {i} {eps:.2e}")


class logger_none(Logger):
    def __call__(self, i, eps, watch=None, **kwargs):
        pass


class MonitorBasic(object):
    """Class for monitoring whether the solver has converged

    Selected attributes
    ---------
    dot_op: operator
        the method to calculate the residual
    iter_max: float/int
        the maximum number of iterations for the solver
    eps_min: float
        the threshold for converge
    """

    def __init__(self, dot_op, iter_max=np.inf, eps_min=1.0e-10, cd_logger=None):
        self.dot_op = dot_op
        self.iter_max = iter_max
        self.eps_min = eps_min
        if cd_logger is None:
            cd_logger = logger_basic()
        self.logger = cd_logger
        self.watch = cinv_utils.StopWatch()
        self.d0 = None
        self.eps = None

    def criterion(self, i, soltn, resid):
        delta = self.dot_op(resid, resid)
        if i == 0:
            self.d0 = delta
            self.eps = []
        else:
            pass

        eps = np.sqrt(delta / self.d0)
        self.eps.append(eps)

        if self.logger is not None:
            self.logger(i, eps, watch=self.watch, soltn=soltn, resid=resid)

        if (i >= self.iter_max) or (delta <= self.eps_min**2 * self.d0):
            return True

        return False

    def __call__(self, *args):
        return self.criterion(*args)

import os,sys
import numpy as np
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
import cinv_utils

## monitors
logger_basic = lambda iter, eps, watch=None, **kwargs: sys.stdout.write(
    "[" + str(watch.elapsed()) + "] " + str((iter, eps)) + "\n"
)
logger_none = lambda iter, eps, watch=None, **kwargs: 0


class MonitorBasic(object):
    """ Class for monitoring whether the solver has converged

    Selected attributes
    ---------
    dot_op: operator
        the method to calculate the residual
    iter_max: float/int
        the maximum number of iterations for the solver
    eps_min: float
        the threshold for converge
    """
    def __init__(
        self, dot_op, iter_max=np.inf, eps_min=1.0e-10, logger=logger_basic
    ):
        self.dot_op = dot_op
        self.iter_max = iter_max
        self.eps_min = eps_min
        self.logger = logger

        self.watch = cinv_utils.StopWatch()

    def criterion(self, iter, soltn, resid):
        delta = self.dot_op(resid, resid)

        if iter == 0:
            self.d0 = delta

        if self.logger is not None:
            self.logger(
                iter,
                np.sqrt(delta / self.d0),
                watch=self.watch,
                soltn=soltn,
                resid=resid,
            )

        if (iter >= self.iter_max) or (delta <= self.eps_min ** 2 * self.d0):
            return True

        return False

    def __call__(self, *args):
        return self.criterion(*args)

"""Logging utilities."""

import logging
import sys

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0


class MPIFilter(logging.Filter):
    """Allow records on root, from healqest, or when record.extra['force'] is True."""

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        if self.rank == 0:
            return True
        # allow explicit override per-call: logger.warning(..., extra={'force': True})
        if getattr(record, "force", False):
            return True
        return False


class MPIAwareFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[37m",  # white
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }
    RESET = "\033[0m"
    FAINT = "\033[2;37m"

    def __init__(self, fmt=None, datefmt=None, style='{', use_colors: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.use_colors = use_colors

    def format(self, record):
        # Fixed-width module name
        name = record.name.split(".")[-1][:8].rjust(8)
        name_colored = f"{self.FAINT}{name}{self.RESET}" if self.use_colors else name

        rank_str = f"{rank:2d}"
        rank_colored = f"{self.FAINT}{rank_str}{self.RESET}" if self.use_colors else rank_str

        # Time colored by log level
        asctime = self.formatTime(record, self.datefmt)
        time_color = self.COLORS.get(record.levelno, "")
        asctime_colored = f"{time_color}{asctime}{self.RESET}" if self.use_colors else asctime

        # Actual log message
        message = record.getMessage()

        return f"{rank_colored}|{asctime_colored}|{name_colored} {message}"


def verbose2level(verbosity: int) -> int:
    if verbosity is None:
        return None
    verbose = int(verbosity)
    verbose = max(verbose, 0)
    verbose = min(verbose, 4)
    return {0: logging.CRITICAL, 1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}[
        verbose
    ]


def set_verbose(level1, level2=logging.WARNING):
    # suppress low-level logs from other modules, like healpy.
    for name in logging.root.manager.loggerDict:
        if not name.startswith("healqest") and name != "__main__":
            if level2 is not None:
                logging.getLogger(name).setLevel(level2)
        else:
            if level1 is not None:
                logging.getLogger(name).setLevel(level1)


def setup_logger(verbose=None, force=True, verbose_other=2):
    """
    Central logging config. Call this early in your script.

    Parameters
    ----------
    verbose: int=3
        verboisity. 0=CRITICAL, 1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG
    force: bool=True
        If True, removes existing handlers.
    verbose_other: int=2
        set the verbosity of non-healqest packages, e.g., healpy.
    """
    fmt = "[{rank}]{asctime}|{name} {message}"
    datefmt = "%H:%M:%S"

    formatter = MPIAwareFormatter(fmt, datefmt, style='{')

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(MPIFilter(rank))
    # Configure root logger
    if verbose is not None:
        logging.basicConfig(level=verbose2level(verbose), handlers=[handler], force=force)
    set_verbose(verbose2level(verbose), verbose2level(verbose_other))


def get_logger(name: str = None, verbose=None, verbose_other=2) -> logging.Logger:
    """
    Wrapper of logging.getLogger.

    Parameters
    ----------
    name: str=None
        Logger name. If None, returns the "healqest" logger.
    verbose: int=3
        Verbosity level for this logger. 0=CRITICAL, 1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG
    verbose_other: int=2
        Verbosity level for other loggers (e.g., healpy).

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("info that only appears on rank 0")
    >>> logger.warning("warning that only appears on any rank", extra={"force": True})
    """
    set_verbose(verbose2level(verbose), verbose2level(verbose_other))
    if not name:
        return logging.getLogger("healqest")
    return logging.getLogger(name)


def test():
    logger = get_logger(__name__)
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')

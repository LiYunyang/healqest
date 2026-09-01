"""Optional MPI context with a notebook-safe serial fallback."""

import atexit
import builtins
import os

comm = None
rank = 0
size = 1
is_mpi = False
mpi_error = None


def _in_notebook():
    get_ipython = getattr(builtins, "get_ipython", None)
    if get_ipython is None:
        return False
    try:
        return getattr(get_ipython(), "kernel", None) is not None
    except Exception:
        return False


def _has_mpi_context():
    markers = (
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
        ("PMI_RANK", "PMI_SIZE"),
        ("PMI2_RANK", "PMI2_SIZE"),
        ("PMIX_RANK", "PMIX_SIZE"),
    )
    return any(all(key in os.environ for key in group) for group in markers)


if _has_mpi_context() and (not _in_notebook() or os.environ.get("HEALQEST_MPI_IN_NOTEBOOK") == "1"):
    try:
        from mpi4py import rc

        rc.initialize = False
        from mpi4py import MPI

        initialized_here = not MPI.Is_initialized()
        if initialized_here:
            MPI.Init_thread(MPI.THREAD_MULTIPLE)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        is_mpi = True

        if initialized_here:
            atexit.register(lambda: not MPI.Is_finalized() and MPI.Finalize())
    except Exception as exc:
        mpi_error = exc

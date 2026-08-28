"""Compute and save to disk the lensing power spectra."""

import tempfile
import os
import shutil

import healpy as hp
import numpy as np

from mpi4py.MPI import COMM_WORLD as comm
from healqest import startup, log
from healqest.spectrum import KappaMap, compute_ps, ClsDB

logger = log.get_logger(__name__)


def stype2ktypes(stype):
    k1 = stype[:2]
    k2 = stype[2:]
    if not k2:
        k2 = None
    return k1, k2


def stypes2ktypes(spec_types):
    """Convert 4-letter spec types to non-repeated 2-letter kmap types."""
    ktypes = []
    for stype in spec_types:
        k1, k2 = stype2ktypes(stype)
        ktypes.append(k1)
        ktypes.append(k2)
    return set(ktypes)


def get_kmap_and_spec(config, stypes, i, mvtype, N1, mf_pair, curl=False, cmbset='a', skip=False):
    """
    Prepare the kappa maps for all the spectra, and compute the spectra of given types.

    The result will be broadcasted to rank 0 and written to the database there.

    Parameters
    ----------
    config: Config
    stypes: list of str
        list of 4-letter spec types, e.g., 'xxxx', 'xyyx'
    i: int
        seed number
    mvtype: str
        the lensrec MV type, e.g., "TT", "EB"
    N1: bool
        whether to compute N1-type spectra. if False, compute N0-type spectra.
    mf_pair: tuple of int
        the meanf-field group for the two kappa maps.
    curl: bool
        whether to compute the curl mode spectra.
    cmbset: str='a'
        the CMB set for the sims, e.g., 'a', 'b', '
    skip: bool
        whether to skip the spectra that already exist in the database.
    """
    # validate the single database for this call
    db = None
    to_skip = list()

    for stype in stypes:
        k1, k2 = stype2ktypes(stype)
        db_new, sql_key = config.get_sql_keys(
            tag=mvtype, seed=i, ktype1=k1, ktype2=k2, N1=N1, SAN0=False, cmbset=cmbset, curl=curl
        )
        if db is None:
            db = db_new
        else:
            assert db == db_new, "all specs in the same call should write to the same SQLite table."
        if skip:
            if db.query(sql_key, return_data=False):
                spec_key = '_'.join(sql_key.values())
                logger.info(f"skipping {db.name} {db.table}: {spec_key}", extra={'force': True})
                to_skip.append(stype)
    stypes = [s for s in stypes if s not in to_skip]

    with tempfile.TemporaryDirectory(prefix='lens', dir=config.tmp_dir) as tmp:
        kmaps = dict()
        for ktype in stypes2ktypes(stypes):
            for g in set(mf_pair):
                kmaps[(ktype, g)] = KappaMap(
                    config, i, ktype, mvtype=mvtype, mf_group=g, N1=N1, cmbset=cmbset, outdir=tmp, curl=curl
                )

        local_results = []
        for stype in stypes:
            k1, k2 = stype2ktypes(stype)
            g1, g2 = mf_pair
            _, sql_key = config.get_sql_keys(
                tag=mvtype, seed=i, ktype1=k1, ktype2=k2, N1=N1, SAN0=False, cmbset=cmbset, curl=curl
            )
            cl_dat = compute_ps(kmaps[(k1, g1)], kmaps[(k2, g2)])
            local_results.append((sql_key, cl_dat))
        if local_results:
            comm.send((db.path, db.table, local_results), dest=comm.size - 1)


def build_task_loop(args, config):
    """Build the requested standard, RDN0, and N1 spectrum tasks."""
    tasks = []

    if args.std:
        seeds = np.arange(config.sim_range[0], config.sim_range[1] + 1)
        tasks.extend((seed, args.set, 'std') for seed in seeds)

    if args.rdn0:
        seeds = np.arange(config.sim_range[0], config.sim_range[1] + 1)
        tasks.extend((seed, args.set, 'rdn0') for seed in np.unique(np.concatenate(([0], seeds))))

    if args.n1:
        seeds = np.arange(config.sim_range_N1[0], config.sim_range_N1[1] + 1)
        assert 0 not in seeds, "N1-type spectra should not include seed0 (data)!"
        tasks.extend((seed, 'a', 'n1') for seed in seeds)

    if not tasks:
        raise ValueError("select at least one spectrum mode: -std, -rdn0, or -n1")
    return tasks


def main(i, mvtype, cmbset, mode, curl=False, skip=False):
    mf_pair = [1, 2] if config.mfsplit else [0, 0]
    common_kw = dict(i=i, mvtype=mvtype, skip=skip, curl=curl, config=config)
    if mode == 'std':
        stypes = ['xxxx', 'xyyx', 'xyxy']
        get_kmap_and_spec(stypes=stypes, N1=False, cmbset=cmbset, mf_pair=mf_pair, **common_kw)
        if args.cross:
            get_kmap_and_spec(stypes=['xx'], N1=False, cmbset=cmbset, mf_pair=(0, 0), **common_kw)

    elif mode == 'rdn0':
        stypes = ['xxxx'] if i == 0 else ['x0x0', 'x00x', '0xx0', '0x0x']
        get_kmap_and_spec(stypes=stypes, N1=False, cmbset=cmbset, mf_pair=mf_pair, **common_kw)

    elif mode == 'n1':
        stypes = ['abba', 'abab', 'xyxy', 'xyyx', 'aabb']
        get_kmap_and_spec(stypes=stypes, N1=True, cmbset='a', mf_pair=mf_pair, **common_kw)
        if args.cross:
            get_kmap_and_spec(stypes=['aa'], N1=True, cmbset='a', mf_pair=(0, 0), **common_kw)
    else:
        raise ValueError(f"unknown spectrum mode: {mode}")


if __name__ == "__main__":
    """
    Compute power spectra of all lensing reconstruction.

    Prerequisites
    -------------
    - lensing reconstruction maps, generated by `rec_lens.py`.
    - meanf-field files generated by `get_plmstack.py`.

    Examples
    --------
    - standard N0-type spectra
    >>> $run scripts/compute_cls.py -c $config -f $field -mvtype $mv -std [-curl]
    - N1-type spectra
    >>> $run scripts/compute_cls.py -c $config -f $field -mvtype $mv -n1 [-curl]
    - RDN0-type spectra, including the seed-0 data spectrum
    >>> $run scripts/compute_cls.py -c $config -f $field -mvtype $mv -rdn0 [-curl]
    """
    parser = startup.parser()
    parser.add_argument('-std', action='store_true', help='do standard Cls')
    parser.add_argument('-rdn0', action='store_true', help='do RDN0-type operations')
    parser.add_argument('-mvtype', default=None, type=str, help='MV type')
    parser.add_argument('-cross', action='store_true', help='compute cross spectra')
    parser.add_argument('-curl', action='store_true', help='compute the curl mode')
    parser.add_argument('-set', default='a', type=str, help='cmbset for std/N0-type sims')
    args = parser.parse_args()
    log.setup_logger(verbose=args.verbose)
    config = startup.Config.from_args(args)

    assert comm.size > 1, f"{__name__} only works in MPI mode."

    config.tmp_dir = config.path(config.outdir, 'tmp/')  # /tmp might be too small for storage
    config.tmp_file_mask = os.path.join(config.tmp_dir, 'psmask.fits')

    if comm.rank == 0:
        os.makedirs(config.tmp_dir, exist_ok=True)
        hp.write_map(config.tmp_file_mask, config.mask_ps, dtype=np.float32, overwrite=True)
    comm.barrier()

    try:
        task_loop = build_task_loop(args, config)
    except ValueError as exc:
        parser.error(str(exc))

    if comm.rank == comm.size - 1:
        ClsDB.mpi_write(comm)
    else:
        for _i, _cmbset, _mode in task_loop[comm.rank :: (comm.size - 1)]:
            main(_i, cmbset=_cmbset, mode=_mode, mvtype=args.mvtype, curl=args.curl, skip=args.skip)
        comm.send(None, dest=comm.size - 1)

    comm.barrier()
    if comm.rank == 0:
        os.unlink(config.tmp_file_mask)
        shutil.rmtree(config.tmp_dir)

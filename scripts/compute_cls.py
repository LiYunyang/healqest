"""Compute and save to disk the lensing power spectra."""

import tempfile
import os
import shutil

import healpy as hp
import numpy as np

from mpi4py.MPI import COMM_WORLD as comm
from healqest import startup, healqest_utils as hq, log
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


def get_kmap_and_spec(  # noqa: C901
    config,
    stypes,
    i,
    mvtype,
    N1,
    mf_pair,
    spectype,
    split=None,
    curl=False,
    cmbset='a',
    skip=False,
    copies=None,
):
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
    spectype: str
        the spectrum type for the database table, e.g., 'n0', 'n1', 'rdn0'.
    split: str, optional
        Data split selecting the power-spectrum mask and output database.
    curl: bool
        whether to compute the curl mode spectra.
    cmbset: str='a'
        the CMB set for the sims, e.g., 'a', 'b', '
    skip: bool
        whether to skip the spectra that already exist in the database.
    copies: dict[str, list[str]], optional
        Mapping of each canonical spectrum type to its duplicate output spectrum types.

    """
    copies = {} if copies is None else copies
    stypes_src = list(stypes)
    for stype_src, stypes_cpy in copies.items():
        if stype_src not in stypes_src:
            raise ValueError(f"copy source {stype_src} is not in stypes")
        if isinstance(stypes_cpy, str):
            raise TypeError(f"copies[{stype_src}] must be a list of spectrum types")

    stypes_out_by_src = {stype_src: [stype_src, *copies.get(stype_src, [])] for stype_src in stypes_src}
    stypes_all = [stype for stypes_out in stypes_out_by_src.values() for stype in stypes_out]
    if len(stypes_all) != len(set(stypes_all)):
        raise ValueError("a spectrum type can only be written once")

    sql_keys = {}
    existing = {}
    db = config.get_sql_table(mvtype, spec_type=spectype, curl=curl, split=split)
    for _stype in stypes_all:
        k1, k2 = stype2ktypes(_stype)
        sql_key = config.get_sql_keys(seed=i, ktype1=k1, ktype2=k2, cmbset=cmbset)
        sql_keys[_stype] = sql_key
        existing[_stype] = skip and db.query(sql_key, return_data=False)

    stypes_src_run = []
    for stype_src, stypes_out in stypes_out_by_src.items():
        if skip and all(existing[stype_out] for stype_out in stypes_out):
            keys = ['_'.join(sql_keys[_stype].values()) for _stype in stypes_out]
            logger.info(f"skipping {db.name} {db.table}: {keys}", extra={'force': True})
            continue
        stypes_src_run.append(stype_src)

    with tempfile.TemporaryDirectory(prefix='lens', dir=config.tmp_dir) as tmp:
        kmaps = dict()
        for ktype in stypes2ktypes(stypes_src_run):
            for g in set(mf_pair):
                kmaps[(ktype, g)] = KappaMap(
                    config,
                    i,
                    ktype,
                    mvtype=mvtype,
                    mf_group=g,
                    N1=N1,
                    cmbset=cmbset,
                    outdir=tmp,
                    curl=curl,
                    split=split,
                )

        local_results = []
        for stype_src in stypes_src_run:
            k1, k2 = stype2ktypes(stype_src)
            g1, g2 = mf_pair
            cl_dat = compute_ps(kmaps[(k1, g1)], kmaps[(k2, g2)])
            for stype_out in stypes_out_by_src[stype_src]:
                if existing[stype_out]:
                    continue
                if stype_out != stype_src:
                    logger.warning(
                        f"quick mode: copying {mvtype} {stype_src} spectrum to {stype_out}",
                        extra={'force': True},
                    )
                local_results.append((sql_keys[stype_out], cl_dat))
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


def main(i, mvtype, cmbset, mode, split=None, curl=False, skip=False):
    mf_pair = [1, 2] if config.mfsplit else [0, 0]
    common_kw = dict(i=i, mvtype=mvtype, skip=skip, curl=curl, config=config, split=split)
    quick = config.quick and hq.mv_is_symm(mvtype)
    if mode == 'std':
        if not quick:
            stypes = ['xxxx', 'xyyx', 'xyxy']
            copies = None
        else:
            stypes = ['xxxx', 'xyxy']
            copies = {'xyxy': ['xyyx']}
        get_kmap_and_spec(
            stypes=stypes, copies=copies, N1=False, cmbset=cmbset, mf_pair=mf_pair, **common_kw, spectype='n0'
        )
        if args.cross:
            get_kmap_and_spec(
                stypes=['xx'], N1=False, cmbset=cmbset, mf_pair=(0, 0), **common_kw, spectype='n0'
            )

    elif mode == 'rdn0':
        if i == 0:
            stypes = ['xxxx']
            copies = None
        else:
            if not quick:
                stypes = ['x0x0', 'x00x', '0xx0', '0x0x']
                copies = None
            else:
                stypes = ['0x0x']
                copies = {'0x0x': ['x00x', '0xx0', 'x0x0']}
        get_kmap_and_spec(
            stypes=stypes,
            copies=copies,
            N1=False,
            cmbset=cmbset,
            mf_pair=mf_pair,
            **common_kw,
            spectype='rdn0',
        )

    elif mode == 'n1':
        if not quick:
            stypes = ['abba', 'abab', 'xyxy', 'xyyx', 'aabb']
            copies = None
        else:
            stypes = ['abab', 'xyxy', 'aabb']
            copies = {'abab': ['abba'], 'xyxy': ['xyyx']}
        get_kmap_and_spec(
            stypes=stypes, copies=copies, N1=True, cmbset='a', mf_pair=mf_pair, **common_kw, spectype='n1'
        )
        if args.cross:
            get_kmap_and_spec(stypes=['aa'], N1=True, cmbset='a', mf_pair=(0, 0), **common_kw, spectype='n1')
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
    parser.add_argument('-split', default=None, type=startup.none_str, help='Data split')
    args = parser.parse_args()
    log.setup_logger(verbose=args.verbose)
    config = startup.Config.from_args(args)

    assert comm.size > 1, f"{__name__} only works in MPI mode."

    config.tmp_dir = config.path(config.outdir, 'tmp/')  # /tmp might be too small for storage
    config.tmp_file_mask = os.path.join(config.tmp_dir, 'psmask.fits')
    if comm.rank == 0:
        os.makedirs(config.tmp_dir, exist_ok=True)
        hp.write_map(config.tmp_file_mask, config.mask_ps(args.split), dtype=np.float32, overwrite=True)
    comm.barrier()

    try:
        task_loop = build_task_loop(args, config)
    except ValueError as exc:
        parser.error(str(exc))

    if comm.rank == comm.size - 1:
        ClsDB.mpi_write(comm)
    else:
        for _i, _cmbset, _mode in task_loop[comm.rank :: (comm.size - 1)]:
            main(
                _i,
                cmbset=_cmbset,
                mode=_mode,
                mvtype=args.mvtype,
                split=args.split,
                curl=args.curl,
                skip=args.skip,
            )
        comm.send(None, dest=comm.size - 1)

    comm.barrier()
    if comm.rank == 0:
        os.unlink(config.tmp_file_mask)
        shutil.rmtree(config.tmp_dir)

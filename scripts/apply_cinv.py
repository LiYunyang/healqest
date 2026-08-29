import os
import healpy as hp
import numpy as np
from healqest import startup, log
from healqest import healqest_utils as hq

from healqest.cinv import cinv_hp as cinv
from mpi4py.MPI import COMM_WORLD as comm

logger = log.get_logger(__name__)


def main(seed, cmbset, N1, ilc_type):
    fname = config.p_cinv(seed=seed, cmbset=cmbset, ilc_type=ilc_type, N1=N1, bundle=args.bundle, ext='fits')
    if args.skip and os.path.exists(fname):
        try:
            hq.verify_fits(fname, 2)
        except Exception:
            logger.error(f"{fname} exists but is corrupted, redoing it.")
            pass
        else:
            logger.warning(f"Skipping {fname}", extra={"force": True})
            return
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    sims = dm.Data(config=config, N1=N1, ilc_type=ilc_type)
    (ninv_t, ninv_p), ninv_nl = sims.get_ninv()
    nlres = sims.get_nlres(cinv=True)
    add_noise = config.add_noise and not N1

    common_kw = dict(
        lmax=config.cinv_lmax,
        nside=config.nside,
        cl=config.cinv_cls['cmb'],
        nl_res=nlres,
        ellscale=config.ellscale,
        g=config.g,
        mtheta=config.mtheta,
        mmin=config.cinv_mmin,
        ninv_nl=ninv_nl,
    )
    if config.rectype == 'sqe':
        cinv_t = cinv.cinv_t(
            ninv=[ninv_t], tf1d=config.tf1d['t'], bl=config.bl, eps_min=config.eps_t, **common_kw
        )

        cinv_p = cinv.cinv_p(
            ninv=[ninv_p, ninv_p], tf1d=config.tf1d['p'], bl=config.bl, eps_min=config.eps_p, **common_kw
        )

        ivfs = cinv.library_cinv_sTP(sims, cinvt=cinv_t, cinvp=cinv_p, add_noise=add_noise)

        tlmbar = ivfs.get_sim_tlm(seed=seed, cmbset=cmbset, bundle=args.bundle)
        elmbar, blmbar = ivfs.get_sim_eblm(seed=seed, cmbset=cmbset, bundle=args.bundle)
        almbar = np.array([tlmbar, elmbar, blmbar])
    elif config.rectype == 'gmv':
        cinv_tp = cinv.cinv_tp(
            ninv=[ninv_t, ninv_p],
            tf1d=[config.tf1d['t'], config.tf1d['p']],
            bl=[config.bl, config.bl],
            eps_min=max(config.eps_t, config.eps_p),
            **common_kw,
        )

        ivfs = cinv.library_cinv_jTP(sims, cinv_jtp=cinv_tp, add_noise=add_noise)
        almbar = ivfs.get_sim_teblm(seed=seed, cmbset=cmbset, bundle=args.bundle)
    else:
        raise NotImplementedError(f"cinv for {config.rectype} is not implemented")
    mapbar = np.atleast_2d(config.g.alm2map(almbar, lmax=config.cinv_lmax))
    mapbar[:, config.mask_boundary == 0] = hp.UNSEEN
    fl = np.array([ivfs.get_fl('t'), ivfs.get_fl('e'), ivfs.get_fl('b'), ivfs.get_fl('te')])
    hq.cinv_io(fname, mapbar, fl=fl, eps=ivfs.get_eps())


def build_task_loop(args, config):
    """Build the requested standard, RDN0, and N1 CINV tasks."""
    tasks = []

    if args.std:
        sim_range = np.arange(config.sim_range[0], config.sim_range[1] + 2)
        tasks.extend((seed, args.set, False, ilc_type) for seed in sim_range for ilc_type in args.ilc)

    if args.rdn0:
        tasks.extend((0, args.set, False, ilc_type) for ilc_type in args.ilc)

    if args.n1:
        seeds = np.arange(config.sim_range_N1[0], config.sim_range_N1[1] + 2)
        assert 0 not in seeds, "N1-type CINV should not include seed0 (data)!"
        tasks.extend(
            (seed, cmbset, True, ilc_type) for seed in seeds for cmbset in 'ab' for ilc_type in args.ilc
        )

    if not tasks:
        raise ValueError("select at least one CINV mode: -std, -rdn0, or -n1")
    return tasks


if __name__ == "__main__":
    """
    Prepare cinv-filtered maps.

    Prerequisites
    -------------
    None

    Examples
    --------
    - standard set-a sims
    >>> $run scripts/apply_cinv.py -c $config -m $data -f $field -std [-set a] -skip -ilc mv

    - seed-0 CINV for RDN0 reconstruction
    >>> $run scripts/apply_cinv.py -c $config -m $data -f $field -rdn0 [-set a] -skip -ilc mv

    - N1 set-a/b sims
    >>> $run scripts/apply_cinv.py -c $config -m $data -f $field -n1 -skip -ilc mv
    """

    parser = startup.parser()
    parser.add_argument('-std', action='store_true', help='do standard/N0-type operations')
    parser.add_argument('-rdn0', action='store_true', help='do seed-0 CINV for RDN0-type operations')
    parser.add_argument('-ilc', nargs='+', default=['mv'], type=str, help='ILC type(s)')
    parser.add_argument('-set', default='a', type=str, help='cmbset for std/N0-type sims')
    parser.add_argument(
        "-m",
        "--module_path",
        required=True,
        help="Path to the data module script (e.g., data.ilc.py) that can prepare data/sims and "
        "auxiliary files (nlres, ninv) for filtering inputs.",
    )
    args = parser.parse_args()
    dm = hq.load_module("healqest.data_module", args.module_path)

    log.setup_logger(verbose=args.verbose)
    config = startup.Config.from_args(args)

    try:
        task_loop = build_task_loop(args, config)
    except ValueError as exc:
        parser.error(str(exc))

    for seed, cmbset, N1, ilc_type in task_loop[comm.rank :: comm.size]:
        main(seed=seed, cmbset=cmbset, N1=N1, ilc_type=ilc_type)

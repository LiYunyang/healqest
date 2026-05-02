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
    ninv_t, ninv_p = sims.get_ninv()
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
    )
    if config.rectype == 'sqe':
        cinv_t = cinv.cinv_t(
            ninv=[ninv_t],
            tf1d=config.tfbl_1d('t'),
            tf2d=config.tfbl_2d('t'),
            bl=config.bl,
            eps_min=config.eps_t,
            **common_kw,
        )

        cinv_p = cinv.cinv_p(
            ninv=[ninv_p, ninv_p],
            tf1d=config.tfbl_1d('p'),
            bl=config.bl,
            tf2d=config.tfbl_2d('p'),
            eps_min=config.eps_p,
            **common_kw,
        )

        ivfs = cinv.library_cinv_sTP(sims, cinvt=cinv_t, cinvp=cinv_p, add_noise=add_noise)

        tlmbar = ivfs.get_sim_tlm(seed=seed, cmbset=cmbset, bundle=args.bundle)
        elmbar, blmbar = ivfs.get_sim_eblm(seed=seed, cmbset=cmbset, bundle=args.bundle)
        almbar = np.array([tlmbar, elmbar, blmbar])
    elif config.rectype == 'gmv':
        cinv_tp = cinv.cinv_tp(
            ninv=[ninv_t, ninv_p, ninv_p],
            tf1d=[config.tfbl_1d('t'), config.tfbl_1d('p')],
            tf2d=[config.tfbl_2d('t'), config.tfbl_2d('p')],
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


if __name__ == "__main__":
    """
    Prepare cinv-filtered maps.

    Prerequisites
    -------------
    None

    Examples
    --------
    - standard set-a sims, from seed-0 (data) to seed 100 (for N0, i.e. set-a)
    >>> $run scripts/apply_cinv.py -c $config -m $data -f $field -i1 0 -i2 100 [-set a] -skip

    - standard set-a/b sims, from seed-1 to seed 100 (for N1)
    >>> $run scripts/apply_cinv.py -c $config -m $data -f $field -i1 1 -i2 100 [-set a] -skip -n1
    """

    parser = startup.parser()
    parser.add_argument('-i1', default=1, type=int, help='seed start')
    parser.add_argument('-i2', default=1, type=int, help='seed stop (inclusive)')
    parser.add_argument('-ilc', default='mv', type=str, help='ILC type')
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

    from itertools import product

    seed_loop = np.arange(args.i1, args.i2 + 1)
    if args.n1:
        # do cmbsets a and b
        loop = list(product(seed_loop, ['a', 'b'], config.ilcs))
    else:
        # do cmbset as requested ('a' by default)
        loop = list(product(seed_loop, [args.set], config.ilcs))
    for i, _cmbset, ilc in loop[comm.rank :: comm.size]:
        main(seed=i, cmbset=_cmbset, N1=args.n1, ilc_type=ilc)

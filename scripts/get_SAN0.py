from collections import defaultdict
from functools import lru_cache
from itertools import product
from typing import Optional
import numpy as np
import healpy as hp
from healqest import weights, resp, startup, healqest_utils as hq, qest, log
from healqest.spectrum import ClsDB
from mpi4py.MPI import COMM_WORLD as comm

logger = log.get_logger(__name__)


def get_db(config, mvtype, seed, cmbset, curl):
    return config.get_sql_keys(
        tag=mvtype, seed=seed, ktype1='xx', ktype2='xx', SAN0=True, N1=False, cmbset=cmbset, curl=curl
    )


def main(seed, cmbset, bundle_pair=None):  # noqa: C901
    logger.info(f"compute SAN0: seeds {seed}; bundles {bundle_pair}")

    if bundle_pair is None:
        b1, b2 = None, None
    else:
        b1, b2 = bundle_pair

    if args.skip:
        qes = list()
        mvtypes = list()
        for mvtype in config.mvtypes:
            db, sql_key = get_db(config, mvtype, seed, cmbset, curl=args.curl)
            if db.query(sql_key, return_data=False):
                logger.info(f"skipping {mvtype} for seed {seed}", extra={'force': True})
                continue
            else:
                qes += hq.mvtype2qe(mvtype)
                mvtypes.append(mvtype)
        qes = list(set(qes))
    else:
        qes = config.qes
        mvtypes = config.mvtypes

    if not qes:
        logger.warning(f"No qe needed, skipping SAN0", extra={"force": True})
        return
    else:
        logger.info(f"Performing SAN0: {mvtypes} QE: {qes}")

    def func(cmbset, seed, bundle, ilc_type, as_dict=False):
        _maps, flms = hq.cinv_io(
            config.p_cinv(seed=seed, cmbset=cmbset, ilc_type=ilc_type, N1=False, bundle=bundle)
        )
        _maps[0] *= config.mask_qe['t']
        _maps[1:] *= config.mask_qe['p']
        almbars = config.g.map2alm(_maps, lmax=config.lmax, check=False).astype(np.complex128)

        flms = flms[:, : config.lmax + 1]
        del _maps

        # apply the lmin, lmax selection for QE
        hp.almxfl(almbars[0], config.flT, inplace=True)
        hp.almxfl(almbars[1], config.flP, inplace=True)
        hp.almxfl(almbars[2], config.flP, inplace=True)
        if as_dict:
            almbars = {s: _ for s, _ in zip('TEB', almbars)}
        return almbars, flms

    fsky_tt = np.mean(config.mask_cinv['t'] ** 2)
    fsky_tp = np.mean(config.mask_cinv['t'] * config.mask_cinv['p'])
    fsky_pp = np.mean(config.mask_cinv['p'] ** 2)

    def qe2fsky(qe):
        assert set(qe.lower()).issubset('teb')
        assert len(qe) == 2
        if set(qe.lower()) == {'t'}:
            return fsky_tt
        elif set(qe.lower()).issubset({'e', 'b'}):
            return fsky_pp
        else:
            return fsky_tp

    almbars = dict()
    fls = dict()
    for ilc in config.ilcs:
        almbars[ilc], fls[ilc] = func(cmbset, seed=seed, bundle=b1, as_dict=True, ilc_type=ilc)

    def get_clqq(
        qe1: str,
        qe2: str,
        pair1: tuple,
        pair2: tuple,
        u1: Optional[np.ndarray] = None,
        u2: Optional[np.ndarray] = None,
    ):
        """Get the response for a single pair of ilc combination.

        Parameters
        ----------
        qe1, qe2: str
            TT, MV, GMVph etc.
        pair1, pair2: tuple.
            The ilc pair for the two estimators. e.g., ('mv', 'tszfree')
        u1,u2: np.ndarray, optional.
            The presence of u1,u2 determines if the calculation is for a hardened calculation or not.
            For example, for TT-TTprf, if u2 is provided, then the cross between TT and the source estimator
            is performed. Otherwise, the estimator is dispatched to two calculations:
            `get_clqq('TT', 'TTprf', None, u)` and `get_clqq('TT', 'TT', None, None)` and combined.
        """
        if qe1 in qest.Qest.__PH_ESTIMATORS__ and u1 is None:
            qest1 = get_estimator(pair1[0], pair1[1])
            out = get_clqq(qe1.removesuffix('ph'), qe2, pair1, pair2, u1=None, u2=u2)
            for j, u in enumerate(qest1.u):
                clqq_prf_j = get_clqq('TT', qe2, pair1, pair2, u1=u, u2=u2)
                w = qest1.get_harden_weights(qe1.removesuffix('ph'), j, curl=args.curl)
                out += w * clqq_prf_j
            return out

        if qe2 in qest.Qest.__PH_ESTIMATORS__ and u2 is None:
            qest2 = get_estimator(pair2[0], pair2[1])
            out = get_clqq(qe1, qe2.removesuffix('ph'), pair1, pair2, u1=u1, u2=None)
            for j, u in enumerate(qest2.u):
                clqq_prf_j = get_clqq(qe1, 'TT', pair1, pair2, u1=u1, u2=u)
                w = qest2.get_harden_weights(qe2.removesuffix('ph'), j, curl=args.curl)
                out += w * clqq_prf_j
            return out

        qeXY = weights.WeightsPlus(
            qe1, config.cmbcl, config.lmax, u=u1, curl=args.curl, distortion='lens' if u1 is None else 'prf'
        )
        qeZA = weights.WeightsPlus(
            qe2, config.cmbcl, config.lmax, u=u2, curl=args.curl, distortion='lens' if u2 is None else 'prf'
        )
        X = almbars[pair1[0]][qe1[0]]
        Y = almbars[pair1[1]][qe1[1]]
        Z = almbars[pair2[0]][qe2[0]]
        A = almbars[pair2[1]][qe2[1]]

        XZ = hp.alm2cl(X, Z) / qe2fsky(qe1[0] + qe2[0])
        YA = hp.alm2cl(Y, A) / qe2fsky(qe1[1] + qe2[1])
        XA = hp.alm2cl(X, A) / qe2fsky(qe1[0] + qe2[1])
        YZ = hp.alm2cl(Y, Z) / qe2fsky(qe1[1] + qe2[0])
        ret = np.zeros(config.Lmax + 1, dtype=np.complex128)
        return resp.fill_clq1q2_fullsky(qeXY, qeZA, ret, XZ, YA, XA, YZ, fast=True)

    san0_keys = list()
    for j, mvtype in enumerate(mvtypes):
        for q1 in hq.mvtype2qe(mvtype):
            for q2 in hq.mvtype2qe(mvtype):
                san0_keys.append((q1, q2))
    san0_keys = set(san0_keys)

    logger.info(f"computing san0 from: {san0_keys}")
    clqq = defaultdict(lambda: 0)
    ilc_pair = list(zip(config.ilcs, config.ilcs[::-1]))
    ilc_pair_norm = len(ilc_pair) ** 2

    do_ph = any([qe.endswith('ph') for qe in qes])
    if do_ph:
        assert len(config.ilcs) == 1, "Unlikely you want to do profile-hardening with multiple ilc pairs!"

    @lru_cache(maxsize=None)
    def get_estimator(ilc1, ilc2):
        out = qest.Qest(
            lmax=config.lmax,
            g=config.g,
            Cls=config.cmbcl,
            Lmax=config.Lmax,
            flT=config.flT,
            flP=config.flP,
            fast=True,
            fls=fls[ilc1],
            fls2=fls[ilc2],
        )
        if do_ph:
            out.init_harden(config.profile_u)
        return out

    for p1 in ilc_pair:
        for p2 in ilc_pair:
            for q1, q2 in san0_keys:
                _clqq = get_clqq(q1, q2, p1, p2, u1=None, u2=None)
                clqq[f'{q1}{q2}'] += _clqq / ilc_pair_norm

    # build mv
    l = np.arange(config.Lmax + 1)

    for j, mvtype in enumerate(mvtypes):
        N0 = 0
        for q1 in hq.mvtype2qe(mvtype):
            for q2 in hq.mvtype2qe(mvtype):
                N0 += clqq[f'{q1}{q2}'].real

        file_resp = config.p_resp(tag=mvtype, bundle=bundle_pair)
        aresp = np.load(file_resp)['grad_resp' if not args.curl else 'curl_resp']
        w = l * (l + 1) / aresp / 2
        N0 *= w**2

        db, sql_key = get_db(config, mvtype, seed, cmbset, curl=args.curl)
        comm.send((db.path, db.table, [(sql_key, N0)]), dest=comm.size - 1)


if __name__ == "__main__":
    """
    Compute the Semi-analytic N0 (SAN0) for the lensing reconstruction.

    Prerequisites
    -------------
    - cinv filtered maps (with corresponding `rectype`), generated by `apply_cinv.py`.

    Notes
    -----
    `-m` is usally unnecessary. Only needed for naive cinv (`rectype=naive`).

    Examples
    --------
    - SAN0 for the grad mode. pairing goes from (i1, i1) to (i2, i2)
    >>> $run scripts/get_SAN0.py -c $config [-m $data] -f $field -i1 $i1 -i2 $i2 -skip

    - SAN0 for the curl mode. pairing goes from (i1, i1) to (i2, i2)
    >>> $run scripts/get_SAN0.py -c $config [-m $data] -f $field -i1 $i1 -i2 $i2 -skip -curl
    """
    parser = startup.parser()
    parser.add_argument('-i1', default=1, type=int, help='seed start')
    parser.add_argument('-i2', default=1, type=int, help='seed stop (inclusive)')
    parser.add_argument('-curl', action='store_true', help='compute the curl mode')
    parser.add_argument('-set', default='a', type=str, help='cmbset for std/N0-type sims')
    parser.add_argument(
        "-m",
        "--module_path",
        required=True,
        help="Path to the data module script (e.g., data.ilc.py) that can prepare data/sims and "
        "auxiliary files (nlres, ninv) for filtering inputs.",
    )
    args = parser.parse_args()

    log.setup_logger(verbose=args.verbose)
    config = startup.Config.from_args(args)
    assert comm.size > 1, f"{__name__} only works in MPI mode."

    _loop = np.arange(args.i1, args.i2 + 1)
    if config.nbundle is None or args.bundle is None:
        bundle_pairs = [[None, None]]
    else:
        # assuming slurm always distribute "nbundle" jobs to compute all lensrec
        # do cross-bundle lensrec
        bundle_pairs = np.array_split(config.bundle_pairs, config.nbundle)[::-1][args.bundle]
        # reversing so the higher rank got more data. This is more efficient for slurm
        # (because rank0 might has other jobs)

    meta_loop = list(product(bundle_pairs, _loop))

    if comm.rank == comm.size - 1:
        ClsDB.mpi_write(comm)
    else:
        for _bundle_pair, _seed in meta_loop[comm.rank :: (comm.size - 1)]:
            main(_seed, args.set, bundle_pair=_bundle_pair)
        comm.send(None, dest=comm.size - 1)
    comm.barrier()

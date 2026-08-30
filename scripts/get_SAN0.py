from collections import defaultdict
from functools import lru_cache
from itertools import product
import numpy as np
import healpy as hp
from healqest import weights, resp, startup, healqest_utils as hq, qest, log
from healqest.spectrum import ClsDB
from mpi4py.MPI import COMM_WORLD as comm

logger = log.get_logger(__name__)


def get_db(config, mvtype, seed, cmbset, curl):
    db = config.get_sql_table(tag=mvtype, spec_type='san0', curl=curl)
    keys = config.get_sql_keys(seed=seed, ktype1='xx', ktype2='xx', cmbset=cmbset)
    return db, keys


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

    do_ph = any(qe in qest.Qest.__PH_ESTIMATORS__ for qe in qes)
    # if do_ph:
    #     assert len(config.ilcs) == 1, "Unlikely you want to do profile-hardening with multiple ilc pairs!"

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

    @lru_cache(maxsize=None)
    def get_estimator(ilc_pair: tuple):
        out = qest.Qest(
            lmax=config.lmax,
            g=config.g,
            Cls=config.cmbcl,
            Lmax=config.Lmax,
            flT=config.flT,
            flP=config.flP,
            fast=True,
            fls=fls[ilc_pair[0]],
            fls2=fls[ilc_pair[1]],
        )
        if do_ph:
            out.init_harden(config.profile_u)
        return out

    @lru_cache(maxsize=None)
    def get_clqq(qe1: str, qe2: str, pair1: tuple, pair2: tuple, u1_idx=None, u2_idx=None):
        """Get the response for a single pair of ilc combination.

        Parameters
        ----------
        qe1, qe2: str
            TT, MV, GMVph etc.
        pair1, pair2: tuple.
            The ilc pair for the two estimators. e.g., ('mv', 'tszfree')
        u1_idx,u2_idx: int, optional.
            Profile indices. The presence of u1,u2 determines if the calculation is for a hardened calculation
            or not. For example, for TT-TTprf, if u2 is provided, then the cross between TT and the source
            estimator is performed. Otherwise, the estimator is dispatched to two calculations:
            `get_clqq('TT', 'TTprf', None, j)` and `get_clqq('TT', 'TT', None, None)` and combined.
        """
        _qe1 = qest.Qest.ph2qe(qe1)
        _qe2 = qest.Qest.ph2qe(qe2)
        if _qe1.count('B') + _qe2.count('B') == 1:
            # early skip the response calculation for odd-parity estimators.
            return np.zeros(config.Lmax + 1, dtype=np.complex128)

        if qest.Qest.isph(qe1):
            assert u1_idx is None
            qest1 = get_estimator(pair1)
            out = get_clqq(_qe1, qe2, pair1, pair2, u1_idx=None, u2_idx=u2_idx).copy()
            for j, _u in enumerate(qest1.u):
                clqq_prf_j = get_clqq('TT', qe2, pair1, pair2, u1_idx=j, u2_idx=u2_idx)
                w = qest1.get_harden_weights(_qe1, j, curl=args.curl)
                out += w * clqq_prf_j
            return out

        if qest.Qest.isph(qe2):
            assert u2_idx is None
            qest2 = get_estimator(pair2)
            out = get_clqq(qe1, _qe2, pair1, pair2, u1_idx=u1_idx, u2_idx=None).copy()
            for j, _u in enumerate(qest2.u):
                clqq_prf_j = get_clqq(qe1, 'TT', pair1, pair2, u1_idx=u1_idx, u2_idx=j)
                w = qest2.get_harden_weights(_qe2, j, curl=args.curl)
                out += w * clqq_prf_j
            return out

        u1 = None if u1_idx is None else get_estimator(pair1).u[u1_idx]
        u2 = None if u2_idx is None else get_estimator(pair2).u[u2_idx]
        kw = dict(curl=args.curl, cls=config.cmbcl, lmax=config.lmax)
        qeXY = weights.WeightsPlus(qe1, u=u1, distortion='lens' if u1_idx is None else 'prf', **kw)
        qeZA = weights.WeightsPlus(qe2, u=u2, distortion='lens' if u2_idx is None else 'prf', **kw)
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

    @lru_cache(maxsize=None)
    def master_clqq(qe1, qe2, pair1, pair2):
        return get_clqq(qe1, qe2, pair1, pair2, u1_idx=None, u2_idx=None)

    def canonicalize(qe1, qe2, pair1, pair2):
        # the following permutations should be the same.
        candidates = [
            (qe1, qe2, pair1, pair2),  #    XYZA
            (qe2, qe1, pair2, pair1),  # = ZAYX
            (qest.Qest.inv_qe(qe1), qest.Qest.inv_qe(qe2), pair1[::-1], pair2[::-1]),  # = YXAZ
            (qest.Qest.inv_qe(qe2), qest.Qest.inv_qe(qe1), pair2[::-1], pair1[::-1]),  # = AZYX
        ]
        return min(candidates)

    san0_keys = list()
    # symmetrized the keys to avoid "double counting" the off-diagonal terms. XYZA should be the same as ZAYX.
    for j, mvtype in enumerate(mvtypes):
        for q1 in hq.mvtype2qe(mvtype):
            for q2 in hq.mvtype2qe(mvtype):
                san0_keys.append((q1, q2))
    san0_keys = set(san0_keys)

    logger.info(f"computing san0 from: {san0_keys}")
    clqq = defaultdict(lambda: 0)
    ilc_pair = list(zip(config.ilcs, config.ilcs[::-1]))
    ilc_pair_norm = len(ilc_pair) ** 2

    for p1 in ilc_pair:
        for p2 in ilc_pair:
            for q1, q2 in san0_keys:
                _clqq = master_clqq(*canonicalize(q1, q2, p1, p2))
                clqq[f'{q1}{q2}'] += _clqq / ilc_pair_norm

    # build mv
    l = np.arange(config.Lmax + 1)

    for j, mvtype in enumerate(mvtypes):
        N0 = 0
        for q1 in hq.mvtype2qe(mvtype):
            for q2 in hq.mvtype2qe(mvtype):
                key = f'{q1}{q2}'
                if key not in clqq:
                    key = f'{q2}{q1}'
                N0 += clqq[key].real

        file_resp = config.p_resp(tag=mvtype, bundle=bundle_pair)
        aresp = np.load(file_resp)['grad_resp' if not args.curl else 'curl_resp']
        w = l * (l + 1) / aresp / 2
        N0 *= w**2

        db, sql_key = get_db(config, mvtype, seed, cmbset, curl=args.curl)
        comm.send((db.path, db.table, [(sql_key, N0)]), dest=comm.size - 1)


def build_task_loop(args, config):
    """Build standard SAN0 tasks from the configured simulation range."""
    if config.nbundle is None or args.bundle is None:
        bundle_pairs = [[None, None]]
    else:
        # assuming slurm always distribute "nbundle" jobs to compute all lensrec
        # do cross-bundle lensrec
        bundle_pairs = np.array_split(config.bundle_pairs, config.nbundle)[::-1][args.bundle]
        # reversing so the higher rank got more data. This is more efficient for slurm
        # (because rank0 might has other jobs)
    seeds = np.arange(config.sim_range[0], config.sim_range[1] + 1)
    return list(product(bundle_pairs, seeds))


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
    - SAN0 for the grad mode
    >>> $run scripts/get_SAN0.py -c $config [-m $data] -f $field -std -skip

    - SAN0 for the curl mode
    >>> $run scripts/get_SAN0.py -c $config [-m $data] -f $field -std -skip -curl
    """
    parser = startup.parser()
    parser.add_argument('-curl', action='store_true', help='compute the curl mode')
    parser.add_argument('-set', default='a', type=str, help='cmbset for std/N0-type sims')
    args = parser.parse_args()

    log.setup_logger(verbose=args.verbose)
    config = startup.Config.from_args(args)
    assert comm.size > 1, f"{__name__} only works in MPI mode."

    task_loop = build_task_loop(args, config)

    if comm.rank == comm.size - 1:
        ClsDB.mpi_write(comm)
    else:
        for _bundle_pair, _seed in task_loop[comm.rank :: (comm.size - 1)]:
            main(_seed, args.set, bundle_pair=_bundle_pair)
        comm.send(None, dest=comm.size - 1)
    comm.barrier()

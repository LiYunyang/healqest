from functools import lru_cache
from itertools import product
import numpy as np
import healpy as hp
import os
import logging
from healqest import weights, resp, startup, healqest_utils as hq, qest
from mpi4py.MPI import COMM_WORLD as comm

logger = logging.getLogger(__name__)


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
            cl_out = config.p_cls(mvtype, seed, seed, 'xx', 'xx', SAN0=True, cmbset=cmbset, curl=args.curl)
            if os.path.exists(cl_out):
                logger.warning(f"Skipping {cl_out}")
                continue
            else:
                qes += config.mvtype2qe(mvtype)
                mvtypes.append(mvtype)
        qes = list(set(qes))
    else:
        qes = config.qes
        mvtypes = config.mvtypes

    if not qes:
        logger.warning(f"No qe needed, skipping SAN0")
        return
    else:
        logger.info(f"Performing SAN0: {mvtypes} QE: {qes}")

    def func(cmbset, seed, bundle, ilc_type, as_dict=False):
        if not cinv:
            dm = hq.load_module("healqest.data_module", args.module_path)
            sims = dm.Data(config=config, N1=False, ilc_type=ilc_type)
            almbars, flms = sims.naive_cinv(config, seed=seed, cmbset=cmbset, bundle=bundle, add_noise=True)
        else:
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

    clqq = dict()

    estimator = qest.Qest(
        lmax=config.lmax,
        nside=config.nside,
        Cls=config.cmbcl,
        Lmax=config.Lmax,
        flT=config.flT,
        flP=config.flP,
        gmv=config.rectype == 'gmv',
    )
    fsky_tt = np.mean(config.mask_cinv['t'] ** 2)
    fsky_tp = np.mean(config.mask_cinv['t'] * config.mask_cinv['p'])
    fsky_pp = np.mean(config.mask_cinv['p'] ** 2)

    def qe2fsky(qe):
        assert set(qe.lower()).issubset('teb')
        assert len(qe) == 2
        if set(qe.lower()) == {'t'}:
            return fsky_tt
        elif set(qe.lower()) == {'eb'}:
            return fsky_pp
        else:
            return fsky_tp

    almbars = dict()
    fls = dict()
    for ilc in config.ilcs:
        almbars[ilc], fls[ilc] = func(cmbset, seed=seed, bundle=b1, as_dict=True, ilc_type=ilc)

    def _get_clqq(qe1, qe2, pair1, pair2, u1=None, u2=None):
        """Get the response for a single pair of ilc combination.

        Parameters
        ----------
        qe1, qe2: str
            TTprf, TT,
        pair1, pair2: tuple
            The ilc pair for the two estimators. e.g., ('mv', 'tszfree')
        u1,u2: array-like
            The presence of u1,u2 determines if the calculation is for a hardened calculation or not.
            For example, for TT-TTprf, if u2 is provided, then the cross between TT and the source estimator
            is performed. Otherwise, the estimator is dispatched to two calculations:
            `get_clqq('TT', 'TTprf', None, u)` and `get_clqq('TT', 'TT', None, None)` and combined.
        """

        @lru_cache(maxsize=None)
        def get_weights(qe_hrd, fl1, fl2):
            weight = estimator.get_harden_weights(
                qe_hrd.removesuffix('prf'), fl1, u=config.profile_u, curl=args.curl, fast=True, fls2=fl2
            )[0]
            return weight

        if qe1 in qest.Qest.__prf_estimators__ and u1 is None:
            w = get_weights(qe1, fls[pair1[0]], fls[pair1[1]])
            clqq_cmb = _get_clqq(qe1.removesuffix('prf'), qe2, pair1, pair2, u1=None, u2=u2)
            clqq_src = _get_clqq('TT', qe2, pair1, pair2, u1=config.profile_u, u2=u2)
            return clqq_cmb + w * clqq_src

        if qe2 in qest.Qest.__prf_estimators__ and u2 is None:
            w = get_weights(qe2, fls[pair2[0]], fls[pair2[1]])
            clqq_cmb = _get_clqq(qe1, qe2.removesuffix('prf'), pair1, pair2, u1=u1, u2=None)
            clqq_src = _get_clqq(qe1, 'TT', pair1, pair2, u1=u1, u2=config.profile_u)
            return clqq_cmb + w * clqq_src

        qeXY = weights.weights_plus(
            qe1, config.cmbcl, config.lmax, u=u1, curl=args.curl, distortion='lens' if u1 is None else 'prf'
        )
        qeZA = weights.weights_plus(
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

    def get_clqq(qe1, qe2):
        # looping over the ilc pairs, for xilc and tsz (gradient cleaning)
        out = 0
        ilc_pair = list(zip(config.ilcs, config.ilcs[::-1]))
        for p1 in ilc_pair:
            for p2 in ilc_pair:
                out += _get_clqq(qe1, qe2, p1, p2) / (len(ilc_pair) ** 2)
        return out

    san0_keys = list()
    for j, mvtype in enumerate(mvtypes):
        for q1 in config.mvtype2qe(mvtype):
            for q2 in config.mvtype2qe(mvtype):
                san0_keys.append((q1, q2))
    san0_keys = set(san0_keys)
    logger.info(f"computing san0 from: {san0_keys}")
    for q1, q2 in san0_keys:
        clqq[f'{q1}{q2}'] = get_clqq(q1, q2)

    # build mv
    l = np.arange(config.Lmax + 1)
    SAN0 = dict()

    for j, mvtype in enumerate(mvtypes):
        if mvtype.startswith('q'):
            raise NotImplementedError(f"q-type {mvtype} is not supported yet. Use")
        N0 = 0
        for q1 in config.mvtype2qe(mvtype):
            for q2 in config.mvtype2qe(mvtype):
                N0 += clqq[f'{q1}{q2}'].real

        file_resp = config.p_resp(tag=mvtype, bundle=bundle_pair)
        aresp = np.load(file_resp)['grad_resp' if not args.curl else 'curl_resp']
        w = l * (l + 1) / aresp / 2
        N0 *= w**2
        SAN0[f'{mvtype}'] = N0

        cl_out = config.p_cls(mvtype, seed, seed, 'xx', 'xx', SAN0=True, cmbset=cmbset, curl=args.curl)
        os.makedirs(os.path.dirname(cl_out), exist_ok=True)
        hq.write_cl(cl_out, N0, header=f"# nlmax, ncor, nside = {config.Lmax:8d} {1:8d} {config.g.nside:8d}")


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

    startup.setup_logger(verbose=args.verbose)
    config = startup.Config.from_args(args)

    cinv = config.rectype != 'naive'

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
    for _bundle_pair, _seed in meta_loop[comm.rank :: comm.size]:
        main(_seed, args.set, bundle_pair=_bundle_pair)

from collections import defaultdict
from itertools import product
import numpy as np
import healpy as hp
import os
from healqest import qest, startup, healqest_utils as hq, log
from mpi4py.MPI import COMM_WORLD as comm

logger = log.get_logger(__name__)


def main(seed1, cmbset1, seed2, cmbset2, N1, bundle_pair=None, combination=None):  # noqa: C901
    logger.info(
        f"lensrec {combination}: seeds {seed1, seed2}; cmbset {cmbset1, cmbset2}; "
        f"bundles {bundle_pair} (N1={N1})"
    )

    if bundle_pair is None:
        b1, b2 = None, None
    else:
        b1, b2 = bundle_pair

    mvtypes = list(config.mvtypes)
    if config.quick and combination in {'yx', 'x0', 'ba'}:
        skipped_mvtypes = [mvtype for mvtype in mvtypes if hq.mv_is_symm(mvtype)]
        if skipped_mvtypes:
            logger.warning(
                f"quick mode: skipping symmetric MV types {skipped_mvtypes} for {combination} "
                f"(seeds {seed1, seed2}; cmbset {cmbset1, cmbset2})",
                extra={'force': True},
            )
            mvtypes = [mvtype for mvtype in mvtypes if mvtype not in skipped_mvtypes]

    if args.skip:
        qes = list()
        pending_mvtypes = list()
        for mvtype in mvtypes:
            file_plm = config.p_plm(
                tag=mvtype,
                seed1=seed1,
                cmbset1=cmbset1,
                seed2=seed2,
                cmbset2=cmbset2,
                N1=N1,
                bundle=bundle_pair,
            )
            if os.path.exists(file_plm):
                try:
                    hq.verify_npy(file_plm)
                except Exception:
                    logger.error(f"{file_plm} exists but is corrupted, redoing it.", extra={"force": True})
                else:
                    logger.warning(f"skipping {mvtype}: {os.path.basename(file_plm)}", extra={"force": True})
                    continue
            qes += hq.mvtype2qe(mvtype)
            pending_mvtypes.append(mvtype)
        mvtypes = pending_mvtypes
        qes = list(set(qes))
    else:
        qes = list(set(qe for mvtype in mvtypes for qe in hq.mvtype2qe(mvtype)))

    if not qes:
        logger.warning(
            f"no qe needed, skipping lensrec {seed1}{cmbset1}{seed2}{cmbset2}", extra={"force": True}
        )
        return

    logger.info(f"Performing MV: {mvtypes} QE: {qes}")

    def func(cmbset, seed, bundle, ilc_type, as_dict=False):
        _maps, flms = hq.cinv_io(
            config.p_cinv(seed=seed, cmbset=cmbset, ilc_type=ilc_type, N1=N1, bundle=bundle)
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

    alms_grads = defaultdict(lambda: 0)
    alms_curls = defaultdict(lambda: 0)
    aresp_grads = defaultdict(lambda: 0)
    aresp_curls = defaultdict(lambda: 0)

    ilc_pair = list(zip(config.ilcs, config.ilcs[::-1]))
    ilc_norm = len(ilc_pair)

    do_ph = any(qe in qest.Qest.__PH_ESTIMATORS__ for qe in qes)
    quick = config.quick and seed1 == seed2 and cmbset1 == cmbset2 and b1 == b2

    def add_reconstruction(qe, glm, clm, aresp_g, aresp_c):
        alms_grads[qe] += glm / ilc_norm
        alms_curls[qe] += clm / ilc_norm
        aresp_grads[qe] += aresp_g / ilc_norm
        aresp_curls[qe] += aresp_c / ilc_norm

    for ilc1, ilc2 in ilc_pair:
        almbars1, flms1 = func(cmbset1, seed1, b1, ilc1)
        almbars2, flms2 = func(cmbset2, seed2, b2, ilc2)

        estimator = qest.Qest(
            lmax=config.lmax,
            g=config.g,
            Cls=config.cmbcl,
            Lmax=config.Lmax,
            flT=config.flT,
            flP=config.flP,
            fast=True,
            fls=flms1,
            fls2=flms2,
        )
        if do_ph:
            estimator.init_harden(config.profile_u, almbars1, almbars2)

        for qe in qes:
            _qe = qest.Qest.ph2qe(qe)
            inv_qe = qest.Qest.inv_qe(qe)
            qe_key = qe, ilc1, ilc2
            inv_qe_key = inv_qe, ilc2, ilc1
            if quick and inv_qe in qes and inv_qe_key < qe_key:
                continue

            (glm, clm), (aresp_g, aresp_c) = estimator.rec_and_resp(_qe, almbars1, almbars2, type1='lens')
            if qest.Qest.isph(qe):
                glm, aresp_g = estimator.profile_harden(_qe, glm, aresp_g, type1='lens', curl=False)
                if config.harden_curl:
                    clm, aresp_c = estimator.profile_harden(_qe, clm, aresp_c, type1='lens', curl=True)
            # Reversed QEs with reversed ILC legs are identical for these inputs.
            add_reconstruction(qe, glm, clm, aresp_g, aresp_c)
            if quick and inv_qe in qes and inv_qe_key != qe_key:
                add_reconstruction(inv_qe, glm, clm, aresp_g, aresp_c)

        del glm, clm, aresp_g, aresp_c, estimator, almbars1, almbars2, flms1, flms2

    # create the common partial index file
    partial_index = np.where(config.mask_boundary > 0)[0]
    index_file = config.p_index
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    if os.path.exists(index_file):
        assert np.all(np.load(index_file)['index'] == partial_index)
    else:
        if comm.rank == 0:
            temp_file = os.path.splitext(index_file)[0] + f'tmp.b{bundle_pair}.npz'
            np.savez(temp_file, index=partial_index.astype(np.uint32), nside=config.nside)
            os.rename(temp_file, index_file)

    # build mv
    for j, mvtype in enumerate(mvtypes):
        glm, aresp_grad = qest.coadd_qe_alm(mvtype, alms_grads, aresp_grads, config.Lmax)
        clm, aresp_curl = qest.coadd_qe_alm(mvtype, alms_curls, aresp_curls, config.Lmax)

        maps = config.g.alm2map([glm, clm], pol=False).astype(np.float32)
        file_plm = config.p_plm(
            tag=mvtype, seed1=seed1, cmbset1=cmbset1, seed2=seed2, cmbset2=cmbset2, N1=N1, bundle=bundle_pair
        )
        os.makedirs(os.path.dirname(file_plm), exist_ok=True)
        np.save(file_plm, maps[:, partial_index])

        if seed1 == seed2 == 1:
            file_resp = config.p_resp(tag=mvtype, bundle=bundle_pair)
            np.savez(file_resp, grad_resp=aresp_grad, curl_resp=aresp_curl)


def expand_loops(loops):
    """Convert (cmbset, seed, cmbset, seed, combination) loops into MPI jobs."""
    out = []
    for cmbset1, seeds1, cmbset2, seeds2, combination in loops:
        assert len(seeds1) == len(seeds2)
        out.append(
            np.array(
                [
                    np.full(len(seeds1), cmbset1),
                    seeds1,
                    np.full(len(seeds2), cmbset2),
                    seeds2,
                    np.full(len(seeds1), combination),
                ],
                dtype=object,
            ).T
        )
    out = np.concatenate(out)
    out[:, 1] = out[:, 1].astype(int)
    out[:, 3] = out[:, 3].astype(int)
    return out


def build_task_loop(args, config):
    """Build the requested standard, RDN0, and N1 reconstruction tasks."""
    task_loops = []

    if args.std or args.std_xx:
        sim_range = np.arange(config.sim_range[0], config.sim_range[1] + 1)
        std_loops = [[args.set, sim_range, args.set, sim_range, 'xx']]
        if not args.std_xx:
            std_loops.extend(
                [
                    [args.set, sim_range, args.set2, sim_range + 1, 'xy'],
                    [args.set2, sim_range + 1, args.set, sim_range, 'yx'],
                ]
            )
        task_loops.append((False, std_loops))

    if args.rdn0:
        assert args.set == args.set2, "RDN0 requires matching cmbset values"
        sim_range = np.arange(config.sim_range[0], config.sim_range[1] + 1)
        zeros = np.zeros_like(sim_range)
        task_loops.append(
            (
                False,
                [
                    [args.set, np.array([0]), args.set, np.array([0]), 'xx'],
                    [args.set, zeros, args.set2, sim_range, '0x'],
                    [args.set2, sim_range, args.set, zeros, 'x0'],
                ],
            )
        )

    if args.n1:
        sim_range = np.arange(config.sim_range_N1[0], config.sim_range_N1[1] + 1)
        assert 0 not in sim_range, "N1-type lensrec should not include seed0 (data)!"
        task_loops.append(
            (
                True,
                [
                    ['a', sim_range, 'b', sim_range, 'ab'],
                    ['b', sim_range, 'a', sim_range, 'ba'],
                    ['a', sim_range, 'a', sim_range + 1, 'xy'],
                    ['a', sim_range + 1, 'a', sim_range, 'yx'],
                    ['a', sim_range, 'a', sim_range, 'aa'],
                    ['b', sim_range, 'b', sim_range, 'bb'],
                ],
            )
        )

    if not task_loops:
        raise ValueError("select at least one reconstruction mode: -std, -std-xx, -rdn0, or -n1")

    task_loop = np.concatenate([np.insert(expand_loops(loops), 4, N1, axis=1) for N1, loops in task_loops])

    if config.nbundle is None or args.bundle is None:
        bundle_pairs = [[None, None]]
    else:
        # assuming slurm always distribute "nbundle" jobs to compute all lensrec
        # do cross-bundle lensrec
        bundle_pairs = np.array_split(config.bundle_pairs, config.nbundle)[::-1][args.bundle]
        # reversing so the higher rank got more data. This is more efficient for slurm
        # (because rank0 might have other jobs)
    return list(product(bundle_pairs, task_loop))


if __name__ == "__main__":
    """
    Prepare lensrec maps.

    Prerequisites
    -------------
    - cinv filtered maps (with corresponding `rectype`), generated by `apply_cinv.py`.

    Examples
    --------
    - standard/N0-type lensing reconstructions
    >>> $run scripts/rec_lens.py -c $config -m $data -f $field -std -skip

    - standard/N0-type xx-only lensing reconstructions
    >>> $run scripts/rec_lens.py -c $config -m $data -f $field -std-xx -skip

    - N1-type lensing reconstructions
    >>> $run scripts/rec_lens.py -c $config -m $data -f $field -n1 -skip

    - RDN0-type lensing reconstructions
    >>> $run scripts/rec_lens.py -c $config -m $data -f $field -rdn0 -skip
    """
    parser = startup.parser()
    parser.add_argument('-std', action='store_true', help='do standard/N0-type operations')
    parser.add_argument('-std-xx', action='store_true', help='do standard/N0-type xx tasks only')
    parser.add_argument('-rdn0', action='store_true', help='do RDN0-type operations')
    parser.add_argument('-set', default='a', type=str, help='cmbset for std/N0-type sims')
    args = parser.parse_known_args()[0]
    parser.add_argument('-set2', default=args.set, type=str, help='cmbset2 for RDN0-type sims')
    args = parser.parse_args()

    log.setup_logger(verbose=args.verbose)
    config = startup.Config.from_args(args)

    task_loop = build_task_loop(args, config)

    for _bundle_pair, (_cmbset1, _seed1, _cmbset2, _seed2, _N1, _combination) in task_loop[
        comm.rank :: comm.size
    ]:
        main(
            _seed1,
            _cmbset1,
            _seed2,
            _cmbset2,
            N1=bool(_N1),
            bundle_pair=_bundle_pair,
            combination=_combination,
        )

from collections import defaultdict
from itertools import product
import numpy as np
import healpy as hp
import os
from healqest import qest, startup, healqest_utils as hq, log
from mpi4py.MPI import COMM_WORLD as comm

logger = log.get_logger(__name__)


def main(seed1, cmbset1, seed2, cmbset2, N1, bundle_pair=None):  # noqa: C901
    logger.info(f"lensrec: seeds {seed1, seed2}; cmbset {cmbset1, cmbset2}; bundles {bundle_pair} (N1={N1})")

    if bundle_pair is None:
        b1, b2 = None, None
    else:
        b1, b2 = bundle_pair

    if config.save_as_map and args.skip:
        qes = list()
        mvtypes = list()
        for mvtype in config.mvtypes:
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
                logger.warning(f"skipping QE: {mvtype}, existing file: {file_plm}", extra={"force": True})
                continue
            else:
                qes += config.mvtype2qe(mvtype)
                mvtypes.append(mvtype)
        qes = list(set(qes))

    else:
        qes = config.qes
        mvtypes = config.mvtypes

    if not qes:
        logger.warning(
            f"no qe needed, skipping lensrec {seed1}{cmbset1}{seed2}{cmbset2}", extra={"force": True}
        )
        return

    estimator = qest.Qest(
        lmax=config.lmax,
        nside=config.nside,
        Cls=config.cmbcl,
        Lmax=config.Lmax,
        flT=config.flT,
        flP=config.flP,
        gmv=config.rectype == 'gmv',
    )

    logger.info(f"Performing MV: {mvtypes} QE: {qes}")

    def func(cmbset, seed, bundle, ilc_type, as_dict=False):
        if not cinv:
            if ilc_type != 'mv':
                raise NotImplementedError("only mv ilc_type is supported for naive filtering")
            sims = dm.Data(config=config, N1=args.n1, ilc_type=ilc_type)
            almbars, flms = sims.naive_cinv(
                config, seed=seed, cmbset=cmbset, bundle=bundle, add_noise=config.add_noise and not N1
            )
        else:
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
    for ilc1, ilc2 in ilc_pair:
        almbars1, flms1 = func(cmbset1, seed1, b1, ilc1)
        almbars2, flms2 = func(cmbset2, seed2, b2, ilc2)
        # assert np.all(flms1==flms2)

        for qe in qes:
            (glm, clm), (aresp_g, aresp_c) = estimator.rec_and_resp(
                qe,
                almbars1,
                almbars2,
                flms1,
                g=config.g,
                fast=True,
                u=config.profile_u,
                type1='lens',
                fls2=flms2,
            )
            alms_grads[qe] += glm / len(ilc_pair)
            alms_curls[qe] += clm / len(ilc_pair)
            aresp_grads[qe] += aresp_g / len(ilc_pair)
            aresp_curls[qe] += aresp_c / len(ilc_pair)

    if not config.save_as_map:
        for qe in qes:
            file_plm = config.p_plm(
                tag=qe, seed1=seed1, cmbset1=cmbset1, seed2=seed2, cmbset2=cmbset2, N1=N1, bundle=bundle_pair
            )
            os.makedirs(os.path.dirname(file_plm), exist_ok=True)
            np.savez(
                file_plm,
                glm=alms_grads[qe],
                clm=alms_curls[qe],
                grad_resp=aresp_grads[qe],
                curl_resp=aresp_curls[qe],
            )
        return

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
    l = np.arange(config.Lmax + 1)
    for j, mvtype in enumerate(mvtypes):
        glm, clm = 0, 0
        aresp_grad = 0
        aresp_curl = 0

        for qe in config.mvtype2qe(mvtype):
            fac = 1
            if mvtype.startswith('q') and qe in ['TE', 'TB', 'EB']:
                fac = 2
            glm += alms_grads[qe] * fac
            clm += alms_curls[qe] * fac
            aresp_grad += aresp_grads[qe] * fac
            aresp_curl += aresp_curls[qe] * fac

        wg = l * (l + 1) / aresp_grad / 2
        wc = l * (l + 1) / aresp_curl / 2

        glm = np.nan_to_num(hp.almxfl(glm, wg))
        clm = np.nan_to_num(hp.almxfl(clm, wc))

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
    """Convert a lists of loops into a single big loop for MPI."""
    out = list()
    for _ in loops:
        idx, l1, jdx, l2 = _
        out.append([np.full(len(l1), idx), l1, np.full(len(l2), jdx), l2])
    out = np.concatenate(out, axis=1, dtype=object).T
    out[:, 1] = out[:, 1].astype(int)
    out[:, 3] = out[:, 3].astype(int)
    return out


if __name__ == "__main__":
    """
    Prepare lensrec maps.

    Prerequisites
    -------------
    - cinv filtered maps (with corresponding `rectype`), generated by `apply_cinv.py`.

    Examples
    --------
    - lensing reconstions for N0 sims. pairing goes upto (i2, i2+1)
    >>> $run scripts/rec_lens.py -c $config -m $data -f $field -i1 $i1 -i2 $i2 -skip

    - lensing reconstions for N1 sims. pairing goes upto (i2, i2+1)
    >>> $run scripts/rec_lens.py -c $config -m $data -f $field -i1 $i1 -i2 $i2 -n1 -skip

    - lensing reconstions for RDN0 . pairing goes from (0, 0) to (0, i2)
    >>> $run scripts/rec_lens.py -c $config -m $data -f $field -i1 0 -i2 $i2 -skip -rdn0
    """
    parser = startup.parser()
    parser.add_argument('-i1', default=1, type=int, help='seed start')
    parser.add_argument('-i2', default=1, type=int, help='seed stop (inclusive)')
    parser.add_argument('-rdn0', action='store_true', help='do RDN0-type operations')
    parser.add_argument('-set', default='a', type=str, help='cmbset for std/N0-type sims')
    args = parser.parse_known_args()[0]
    parser.add_argument('-set2', default=args.set, type=str, help='cmbset2 for RDN0-type sims')
    parser.add_argument(
        "-m",
        "--module_path",
        required=True,
        help="Path to the data module script (e.g., data.ilc.py) that can prepare data/sims "
        "and auxiliary files (nlres, ninv) for filtering inputs.",
    )
    args = parser.parse_args()
    dm = hq.load_module("healqest.data_module", args.module_path)

    log.setup_logger(verbose=args.verbose)
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
        # (because rank0 might have other jobs)

    if args.n1:
        assert 0 not in _loop, "N1-type lensrec should not include seed0 (data)!"
        loops = [
            ['a', _loop, 'b', _loop],  # a1b1
            ['b', _loop, 'a', _loop],  # b1a1
            ['a', _loop, 'a', _loop + 1],  # xy
            ['a', _loop + 1, 'a', _loop],  # yx
            ['a', _loop, 'a', _loop],  # a1a1, optional for auto resp
            ['b', _loop, 'b', _loop],  # b1b1, optional for auto resp
        ]
        loop = expand_loops(loops)
        meta_loop = list(product(bundle_pairs, loop))
        for _bundle_pair, (_cmbset1, _seed1, _cmbset2, _seed2) in meta_loop[comm.rank :: comm.size]:
            main(_seed1, _cmbset1, _seed2, _cmbset2, N1=True, bundle_pair=_bundle_pair)

    else:
        if args.rdn0:
            assert 0 in _loop
            _loop = np.delete(_loop, 0)
            assert args.set == args.set2
            loops = [
                [args.set, [0], args.set, [0]],  # xx
                [args.set, np.zeros_like(_loop), args.set2, _loop],  # 0x
                [args.set2, _loop, args.set, np.zeros_like(_loop)],  # x0
            ]
        else:
            loops = [
                [args.set, _loop, args.set, _loop],  # xx
                [args.set, _loop, args.set2, _loop + 1],  # xy
                [args.set2, _loop + 1, args.set, _loop],  # yx
            ]
        loop = expand_loops(loops)
        meta_loop = list(product(bundle_pairs, loop))
        for _bundle_pair, (_cmbset1, _seed1, _cmbset2, _seed2) in meta_loop[comm.rank :: comm.size]:
            main(_seed1, _cmbset1, _seed2, _cmbset2, N1=False, bundle_pair=_bundle_pair)

import tempfile
import os
import logging
import shutil

import healpy as hp
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from healqest import startup, healqest_utils as hq

logger = logging.getLogger(__name__)


def get_kfile(i, ktype, cmbset, dir_tmp, mvtype, N1, mf_group, bundle):
    s1, s2, cmbset1, cmbset2 = config.ktype2ij(ktype, i, j=None, cmbset=cmbset)
    fname = config.f_tmp(
        tag=mvtype,
        seed1=s1,
        seed2=s2,
        ktype=ktype,
        N1=N1,
        mf_group=mf_group,
        bundle=bundle,
        cmbset1=cmbset1,
        cmbset2=cmbset2,
        curl=args.curl,
    )
    return os.path.join(dir_tmp, fname)


def computse_ps_single(
    config,
    bundle1,
    bundle2,
    i,
    mvtype,
    mvtype2,
    dir_tmp,
    ktype1,
    ktype2,
    mf_group1,
    mf_group2,
    N1,
    klm2,
    mask2,
    cmbset,
    spice_kwargs=None,
):
    file1 = get_kfile(i, ktype1, cmbset, dir_tmp, mvtype=mvtype, N1=N1, mf_group=mf_group1, bundle=bundle1)
    file2 = get_kfile(i, ktype2, cmbset, dir_tmp, mvtype=mvtype2, N1=N1, mf_group=mf_group2, bundle=bundle2)
    if klm2 is not None:
        file2 = klm2

    if config.save_as_map:
        m1 = file1
        m2 = file2
    else:
        m1 = np.nan_to_num(hp.read_alm(file1))
        if isinstance(file2, str):
            m2 = np.nan_to_num(hp.read_alm(file2))
        else:
            m2 = file2  # for cross correlation
    clkk = hq.kappa_spectrum(m1, m2, mask1=file_mask, mask2=mask2, g=config.g, anafast=False, **spice_kwargs)
    return clkk


def compute_ps(
    mvtype,
    i,
    ktype,
    dir_tmp,
    mvtype2=None,
    N1=False,
    file_mask=None,
    mf_pair=(0, 0),
    spice_kwargs=None,
    klm2=None,
    do_bundle=False,
    cmbset=None,
):
    """
    Compute power spectrum of kappa map (stored in tmp dir).

    Parameters
    ----------
    mvtype: str
        MV type of the first map
    i: int
        index of the simulation. This is a single number, and the kappa map indexes will be figured out from
        it based no the ktype arguments.
    ktype: str
        4-letter QE construction type, e.g. xyxy, x0x0, abab
    dir_tmp: str
        Path to the temporary directory where the kappa maps are stored
    mvtype2: str=None
        MV type of the second map. If None, this will be set to `mvtype`
    N1: bool=False
        If True, use N1-type simulations.
    file_mask: str
        Path to the power spectrum mask.
    mf_pair: list or tuple.
        The mean field groups to use for the two maps.
    spice_kwargs: dict
        Keyword parameters for polspice(`kspice`).
    klm2: str=None
        Path to the second map (a full-sky input kappa map). If set, the result would be cross spectra with
        the input, and the default setting disable the masking of the second map.
    do_bundle: bool=False:
        If True, loop over all bundles. All N(N-1)/2 combinations of *cross* spectra will be constructed and
        stored.
    """
    if mvtype2 is not None:
        name = f"{mvtype}x{mvtype2}"
    else:
        name = f"{mvtype}"
        mvtype2 = mvtype

    ktype1 = ktype[:2]
    if mf_pair is None:
        mf_pair = (0, 0)
    g1, g2 = mf_pair

    # HACK: this is a crude approximation to mask^2, because it depends on the QE type.
    # HACK: this sloppiness is okay because the final resp is based on auto cross correlations.
    fsky_qe2 = config.mask_cinv['t'] * config.mask_cinv['p']
    if klm2 is None:
        ktype2 = ktype[2:]
        mask2 = file_mask
        mask_bias = np.mean(fsky_qe2**2 * config.mask_ps**2) / np.mean(config.mask_ps**2)
    else:
        # cross spectrum
        ktype2 = None
        mask2 = None
        mask_bias = np.mean(fsky_qe2 * config.mask_ps) / np.mean(config.mask_ps)

    cl_out = config.p_cls(
        tag=name, seed1=i, seed2=None, ktype1=ktype1, ktype2=ktype2, N1=N1, ext='dat', cmbset=cmbset, curl=args.curl
    )
    if args.skip and os.path.exists(cl_out):
        logger.warning(f"Skipping {cl_out}")
        return

    out = []
    os.makedirs(os.path.dirname(cl_out), exist_ok=True)

    kw = dict(
        dir_tmp=dir_tmp,
        ktype1=ktype1,
        ktype2=ktype2,
        i=i,
        mf_group1=g1,
        mf_group2=g2,
        N1=N1,
        mask2=mask2,
        spice_kwargs=spice_kwargs,
        klm2=klm2,
        cmbset=cmbset,
        mvtype=mvtype,
        mvtype2=mvtype2,
    )
    if do_bundle:
        m = config.nbundle
        clx = computse_ps_single(config=config, bundle1='X', bundle2="X", **kw)  # first term in Eq 38
        clxj = 0  # second term in Eq 38
        clj = 0  # third term in Eq 38
        for j in range(m):
            cl = computse_ps_single(config=config, bundle1=f'X{j}', bundle2=f"X{j}", **kw)
            clxj += cl
            out.append(cl)
        for bp in config.bundle_pairs:
            cl = computse_ps_single(config=config, bundle1=bp, bundle2=bp, **kw)
            clj += cl
            out.append(cl)

        cl_tot = 4 / (m * (m - 1) * (m - 2) * (m - 3)) * (clx - clxj + clj)
        out.append(cl_tot)

    else:
        clkk = computse_ps_single(config=config, bundle1=None, bundle2=None, **kw)
        out.append(clkk)

    out = np.array(out) / mask_bias
    hq.write_cl(cl_out, out, header=f"# nlmax, ncor, nside = {config.Lmax:8d} {1:8d} {config.g.nside:8d}")
    return out


def get_mf(i, tag, ktype, y=None, N1=False, group=0, bundle=None, cmbset=None):
    assert group in [0, 1, 2]
    file_mf = config.p_plm(tag=tag, stack_type=ktype, N1=N1, bundle=bundle, cmbset=cmbset)
    logger.warning(f"using MF: {os.path.basename(file_mf)}")
    i1, i2 = config.sim_range_N1 if N1 else config.sim_range

    if config.save_as_map:
        # read maps and bad pixels are set to 0
        gctag = 'gmf' if not args.curl else 'cmf'
        field = gctag if group == 0 else f"{gctag}{group}"
        mf, h = hq.read_map(file_mf, h=True, field=field, dtype=np.float64)
        h = dict(h)
        assert h['NSIM'] == (i2 - i1 + 1), f"loaded MF ({h['NSIM']}) is inconsistent with configuration settings!"
        nsim = h[f'NSIM'] if group == 0 else h[f'NSIM{group}']
        split_i = h['SPLITIDX']
    else:
        raise NotImplementedError("fix the grad/curl interface first!")
        loaded = np.load(file_mf)
        mf = loaded['mf'] if group == 0 else loaded[f'mf{group}']
        nsim = loaded['nsim'] if group == 0 else loaded[f'nsim{group}']
        split_i = loaded['split_i']

    if i == 0:
        assert ktype == 'xx'
        mf = mf / nsim
    else:
        if (group == 1 and i < split_i + 1) or (group == 2 and i >= split_i + 1) or group == 0:
            mf = (mf - y) / (nsim - 1)
        else:
            mf = mf / nsim
    return mf


def get_kmap(i, ktype='xx', mvtype='TT', dir_tmp='/tmp', N1=False, mf_group=0, do_bundle=False, cmbset=None):
    """Saves kmap that gets fed into polspice.

    Parameters
    ----------
    i: int
        index of sim
    ktype: str
    mvtype: str
        MV used (TT/EE/TE/TB/EB)
    dir_tmp: str
        Path to the default temporary directory
    N1: bool=False
        If True, construct kappa map for N1-type simulations
    mf_group: int
        Mean field group, one of (0, 1, 2)
    do_bundle: bool=False:
        If True, loop over all bundles.
    cmbset: str
    """
    s1, s2, cmbset1, cmbset2 = config.ktype2ij(ktype, i, j=None, cmbset=cmbset)
    logger.info(f"Constructing kappa map {i}, {ktype}: {s1}{cmbset1},{s2}{cmbset2} for group {mf_group}")
    if do_bundle:
        bundle_loop = config.bundle_pairs
        # the averaged bundle-cross maps for choose-4 estimators (Mat's fast algorithm).
        bundle_avg_maps = {f'X{b}': 0 for b in range(config.nbundle)}
        bundle_avg_maps['X'] = 0
    else:
        bundle_loop = [None]
    for bundle_pair in bundle_loop:
        fname = os.path.join(
            dir_tmp,
            config.f_tmp(
                mvtype,
                seed1=s1,
                seed2=s2,
                ktype=ktype,
                N1=N1,
                mf_group=mf_group,
                bundle=bundle_pair,
                curl=args.curl,
                cmbset1=cmbset1,
                cmbset2=cmbset2,
            ),
        )
        if not config.save_as_map:
            qes = config.mvtype2qe(mvtype)
            kmv = 0
            respmv = 0
            for qe in qes:
                file_plm = config.p_plm(
                    tag=qe, seed1=s1, seed2=s2, cmbset1=cmbset1, cmbset2=cmbset2, N1=N1, bundle=bundle_pair
                )
                y = np.load(file_plm)['glm' if not args.curl else 'clm']
                mf = get_mf(
                    i,
                    tag=qe,
                    ktype=ktype,
                    N1=N1,
                    group=mf_group,
                    y=y,
                    bundle=bundle_pair,
                    cmbset=cmbset if len(cmbset) == 1 else 'a',
                )
                y = y - mf

                # Response
                # MC response
                fresp = config.p_resp(tag=qe)
                resp = np.load(fresp)['resp']
                # analytical response
                aresp = np.load(file_plm)['analytical_resp']
                resp *= aresp
                l = np.arange(len(resp))
                p2k_fac = 0.5 * l * (l + 1)
                kmv += hp.almxfl(y, p2k_fac)
                respmv += resp
            respmv = np.nan_to_num(1 / respmv)
            kmv = hp.almxfl(kmv, respmv)
            hp.write_alm(fname, kmv, overwrite=True)
        else:
            file_plm = config.p_plm(
                tag=mvtype, seed1=s1, seed2=s2, cmbset1=cmbset1, cmbset2=cmbset2, N1=N1, bundle=bundle_pair
            )
            y = hq.read_map(file_plm, field=0 if not args.curl else 1, dtype=np.float64)
            mf = get_mf(
                i,
                tag=mvtype,
                ktype=ktype,
                N1=N1,
                group=mf_group,
                y=y,
                bundle=bundle_pair,
                cmbset=cmbset if len(cmbset) == 1 else 'a',
            )
            kmv = y - mf

            # since y and kmv are read as non-partial maps (bad=0), bad pixs should be set explictly to UNSEEN
            # for efficient partial maps.
            kmv[kmv == 0] = hp.UNSEEN  # important for partial maps.
            hp.write_map(fname, kmv, overwrite=True, dtype=np.float64, partial=True)  # polspice needs float64
        if do_bundle:
            for b in bundle_pair:
                bundle_avg_maps[f'X{b}'] += kmv
                # should divide by `config.nbundle`, but this is taken care in `compute_ps`
            bundle_avg_maps['X'] += kmv
            # should divide by `config.nbundle^2/2`, but this is taken care in `compute_ps`
    if do_bundle:
        for key, kmv in bundle_avg_maps.items():
            fname = os.path.join(
                dir_tmp,
                config.f_tmp(
                    mvtype,
                    seed1=s1,
                    seed2=s2,
                    ktype=ktype,
                    N1=N1,
                    mf_group=mf_group,
                    bundle=key,
                    curl=args.curl,
                    cmbset1=cmbset1,
                    cmbset2=cmbset2,
                ),
            )
            hp.write_map(fname, kmv, overwrite=True, dtype=np.float64, partial=True)  # polspice needs float64


def get_kappa_in(i):
    file_klmin = config.path(config.kappa_in, seed=i)
    ilm = hp.read_alm(file_klmin)
    ilm = hq.reduce_lmax(ilm, lmax=config.Lmax)
    return hp.alm2map(np.nan_to_num(ilm), nside=config.g.nside)


def main_mvtype2(i, kw_ps):
    with tempfile.TemporaryDirectory(prefix='lens_qe2') as tmp:
        get_kmap(i, ktype='xy', mvtype=args.mvtype, dir_tmp=tmp)
        get_kmap(i, ktype='yx', mvtype=args.mvtype2, dir_tmp=tmp)
        get_kmap(i, ktype='xy', mvtype=args.mvtype2, dir_tmp=tmp)
        compute_ps(args.mvtype, i, ktype='xyxy', mvtype2=args.mvtype2, **kw_ps, dir_tmp=tmp)
        compute_ps(args.mvtype, i, ktype='xyyx', mvtype2=args.mvtype2, **kw_ps, dir_tmp=tmp)


def main_std(i, cmbset, kw_ps):
    if i == 0:
        return
    mf_pair = [1, 2] if config.mfsplit else [0, 0]
    do_bundle = config.nbundle is not None
    with tempfile.TemporaryDirectory(prefix='lens_std', dir=tmpdir) as tmp:
        for ktype in ['xy', 'yx', 'xx']:
            for g in set(mf_pair):
                get_kmap(
                    i, ktype=ktype, mvtype=args.mvtype, dir_tmp=tmp, mf_group=g, do_bundle=do_bundle, cmbset=cmbset
                )
        for ktype in ['xxxx', 'xyyx', 'xyxy']:
            compute_ps(
                args.mvtype, i, ktype=ktype, **kw_ps, dir_tmp=tmp, mf_pair=mf_pair, do_bundle=do_bundle, cmbset=cmbset
            )
        if args.cross:
            get_kmap(i, ktype='xx', mvtype=args.mvtype, dir_tmp=tmp, mf_group=0)
            compute_ps(args.mvtype, i, ktype='xx', **kw_ps, dir_tmp=tmp, mf_pair=None, klm2=get_kappa_in(i))


def main_n1(i, kw_ps):
    if i == 0:
        return
    mf_pair = [1, 2] if config.mfsplit else [0, 0]
    with tempfile.TemporaryDirectory(prefix='lens_N1', dir=tmpdir) as tmp:
        for ktype in ['ab', 'ba', 'xy', 'yx', 'aa', 'bb']:
            for g in set(mf_pair):
                get_kmap(i, ktype=ktype, mvtype=args.mvtype, N1=True, dir_tmp=tmp, mf_group=g, cmbset='a')
        for ktype in ['abba', 'abab', 'xyxy', 'xyyx', 'aabb']:
            compute_ps(args.mvtype, i, ktype=ktype, N1=True, **kw_ps, dir_tmp=tmp, mf_pair=mf_pair, cmbset='a')
        if args.cross:
            get_kmap(i, ktype='aa', mvtype=args.mvtype, N1=True, dir_tmp=tmp, mf_group=0, cmbset='a')
            compute_ps(
                args.mvtype,
                i,
                ktype='aa',
                N1=True,
                **kw_ps,
                dir_tmp=tmp,
                mf_pair=None,
                klm2=get_kappa_in(i),
                cmbset='a',
            )


def main_rdn0(i, cmbset, kw_ps):
    mf_pair = [1, 2] if config.mfsplit else [0, 0]
    do_bundle = config.nbundle is not None
    # TODO: `do_bundle` for RDN0 hasn't been implemented (only the data spectrum i==0 has)
    with tempfile.TemporaryDirectory(prefix='lens_rdn0') as tmp:
        for ktype in ['xx'] if i == 0 else ['x0', '0x']:
            for g in set(mf_pair):
                get_kmap(
                    i, ktype=ktype, mvtype=args.mvtype, dir_tmp=tmp, mf_group=g, cmbset=cmbset, do_bundle=do_bundle
                )
        for ktype in ['xxxx'] if i == 0 else ['x0x0', 'x00x', '0xx0', '0x0x']:
            compute_ps(
                args.mvtype, i, ktype=ktype, **kw_ps, dir_tmp=tmp, mf_pair=mf_pair, cmbset=cmbset, do_bundle=do_bundle
            )


def main(i, cmbset):
    kw = dict(spice_kwargs=config.spice_kwargs, file_mask=file_mask)

    if args.mvtype2 is not None:  # calc xspec between two estimators
        main_mvtype2(i, kw_ps=kw)
    else:
        if args.std:
            main_std(i, cmbset, kw_ps=kw)
        elif args.rdn0:
            main_rdn0(i, cmbset, kw_ps=kw)
        elif args.n1:
            main_n1(i, kw_ps=kw)


if __name__ == "__main__":
    """
    Compute power spectra of all lensing reconstruction.

    Prerequisites
    -------------
    - lensing reconstruction maps, generated by `rec_lens.py`.
    - meanf-field files generated by `get_plmstack.py`.

    Examples
    --------
    - standard N0-type spectra (i=0 will be ignored)
    >>> $run scripts/compute_cls.py -c $config -f $field -mvtype $mv -i1 1 -i2 $i2 -std [-curl]
    - N1-type spectra (i=0 will be ignored)
    >>> $run scripts/compute_cls.py -c $config -f $field -mvtype $mv -i1 1 -i2 $i2 -n1 [-curl]
    - RDN0-type (and data) spectra (i=0 is data).
    >>> $run scripts/compute_cls.py -c $config -f $field -mvtype $mv -i1 1 -i2 $i2 -rdn0 [-curl]
    """
    parser = startup.parser()
    parser.add_argument('-i1', default=1, type=int, help='seed start')
    parser.add_argument('-i2', default=1, type=int, help='seed stop')
    parser.add_argument('-std', action='store_true', help='do standard Cls')
    parser.add_argument('-rdn0', action='store_true', help='do RDN0-type operations')
    parser.add_argument('-mvtype', default=None, type=str, help='MV type')
    parser.add_argument('-mvtype2', default=None, dest='mvtype2', help='2nd MV type if xpec')
    parser.add_argument('-cross', action='store_true', help='compute cross spectra')
    parser.add_argument('-curl', action='store_true', help='compute the curl mode')
    parser.add_argument('-set', default='a', type=str, help='cmbset for std/N0-type sims')
    args = parser.parse_args()
    startup.setup_logger(verbose=args.verbose)
    config = startup.Config.from_args(args)

    tmpdir = config.path(config.outdir, 'tmp/')  # /tmp might be too small for storage
    file_mask = os.path.join(tmpdir, 'psmask.fits')
    if comm.rank == 0:
        os.makedirs(tmpdir, exist_ok=True)
        hp.write_map(file_mask, config.mask_ps, dtype=np.float32, overwrite=True)
    comm.barrier()

    loop = np.arange(args.i1, args.i2 + 1)
    for _i in loop[comm.rank :: comm.size]:
        main(_i, cmbset=args.set)
    comm.barrier()
    if comm.rank == 0:
        os.unlink(file_mask)
        shutil.rmtree(tmpdir)

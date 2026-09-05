#!/usr/bin/env python3
import os
import healpy as hp
import numpy as np
from healqest import startup, log
from healqest import healqest_utils as hq

logger = log.get_logger(__name__)


def main(seed, cmbset, ilc_type):
    N1 = False
    fname = config.p_ilc(seed=seed, cmbset=cmbset, ilc_type=ilc_type, N1=N1, bundle=args.bundle, ext='fits')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    sims = dm.Data(config=config, N1=N1, ilc_type=ilc_type)

    m0 = sims.get_tmap(seed, 'a', None, add_noise=True, g=config.g)
    m1 = sims.get_pmap(seed, 'a', None, add_noise=True, g=config.g)

    m = np.array([m0, m1[0], m1[1]])
    m[:, config.mask_boundary == 0] = hp.UNSEEN
    hp.write_map(fname, m, overwrite=True, partial=True, dtype=np.float32)


if __name__ == "__main__":
    """
    Save the ILC map (not needed for the pipeline, but good for checking)

    Prerequisites
    -------------
    None

    Examples
    --------
    - save ILC map for data (seed=0)
    >>> $run scripts/make_ilc.py -c $config -m $data -f $field -ilc mv -seed 0
    """

    parser = startup.parser()
    parser.add_argument('-ilc', nargs='+', default=['mv'], type=str, help='ILC type(s)')
    parser.add_argument('-set', default='a', type=str, help='cmbset for std/N0-type sims')
    parser.add_argument('-seed', default=0, type=int, help='seed number of the ilc map')
    parser.add_argument(
        "-m",
        "--module_path",
        required=True,
        help="Path to the data module script (e.g., data.ilc.py) that can prepare data/sims and "
        "auxiliary files (nlres, ninv) for filtering inputs.",
    )
    args = parser.parse_args()
    assert len(args.ilc) == 1
    dm = hq.load_module("healqest.data_module", args.module_path)
    log.setup_logger(verbose=args.verbose)
    config = startup.Config.from_args(args)
    main(seed=args.seed, cmbset=args.set, ilc_type=args.ilc[0])

import numpy as np
import os
from healqest import spectrum, startup, analysis as ha, log
import GPy

logger = log.get_logger(__name__)


class EmulatorSampler:
    """Generate a reproducible Sobol grid over named parameter bounds.

    Parameters
    ----------
    N : int
        Number of parameter points to retain from the scrambled Sobol grid.
        At most 1024 points are supported.
    parameters : dict[str, tuple[float, float]]
        Lower and upper bounds for each sampled parameter. Parameters are
        stored alphabetically to give the grid a deterministic column order.

    Attributes
    ----------
    grid : np.ndarray
        Sampled parameter values with shape ``(N, len(parameters))``.
    """

    def __init__(self, N, parameters):
        self.N = N

        self.parameters = sorted(list(parameters.keys()))
        self.boundaries = np.array([parameters[k] for k in self.parameters])
        self.grid = self.make_grid(
            Ndim=len(self.parameters),
            Nsamp=self.N,
            boundary_dict=parameters,
            ordered_keys=self.parameters,
            seed=0,
        )

    @staticmethod
    def make_grid(Ndim, Nsamp, boundary_dict, ordered_keys, seed):
        """Create a scaled scrambled Sobol grid.

        Parameters
        ----------
        Ndim : int
            Number of sampled dimensions.
        Nsamp : int
            Number of points to retain from a 1024-point Sobol design.
        boundary_dict : dict[str, tuple[float, float]]
            Bounds keyed by parameter name.
        ordered_keys : sequence[str]
            Parameter names in the desired output-column order.
        seed : int
            Seed used to scramble the Sobol sequence.

        Returns
        -------
        np.ndarray
            Scaled sample points of shape ``(Nsamp, Ndim)``.

        Raises
        ------
        ValueError
            If ``Nsamp`` exceeds 1024.
        """
        if Nsamp > 1024:
            raise ValueError("Nsamp should be <= 1024 for Sobol sampling.")

        from scipy.stats import qmc

        sampler = qmc.Sobol(d=Ndim, seed=seed, scramble=True)
        samples = sampler.random_base2(10)[:Nsamp]  # 1024 total samples

        samples = qmc.scale(
            samples,
            l_bounds=[boundary_dict[k][0] for k in ordered_keys],
            u_bounds=[boundary_dict[k][1] for k in ordered_keys],
        )
        return samples

    def get_parameter(self, i):
        assert 0 <= i < self.N, f"i={i} exceeds the number of foreground samples {self.N}"

        out = {k: self.grid[i, j] for j, k in enumerate(self.parameters)}
        return out


def GP1d(y, var, lengthscale=3):
    """
    1d Gaussian process regression with heteroscedastic noise.

    Parameters
    ----------
    y: np.ndarray
        1d array of data to fit.
    var: np.ndarray
        1d array of variances for each data point.
    lengthscale: float
        Lengthscale of the RBF kernel. The x-axis is defined to be the index of the data points, i.e.,
        x = np.arange(len(y)).

    Returns
    -------
    model: GPy.models.GPHeteroscedasticRegression
        GPy model object with a predict method that returns the mean prediction.
    """
    L = np.arange(y.shape[0]).astype(float)
    y = np.asarray(y, dtype=float)
    var = np.asarray(var, dtype=float)
    assert L.shape == y.shape == var.shape
    X = L[:, None]
    Y = y[:, None]
    kernel = GPy.kern.RBF(input_dim=1, variance=np.var(y), lengthscale=lengthscale)
    model = GPy.models.GPHeteroscedasticRegression(X, Y, kernel=kernel)

    # Fix measurement variance to realization-derived variance
    model[".*het_Gauss.variance"] = var[:, None]
    model[".*het_Gauss.variance"].fix()
    model["rbf.lengthscale"].fix()
    model.optimize(optimizer="lbfgs", messages=False)
    _predict = model.predict

    def predict():
        return _predict(np.arange(len(var))[:, None], include_likelihood=False)[0][:, 0]

    model.predict = predict
    return model


def RBFSmooth(var, lengthscale, variance):
    """
    Smoothing matrix of a 1d vector with a RBF kernel.

    Parameters
    ----------
    var: np.ndarray
        Variance of the input vector.
    lengthscale: float
        Lengthscale of the RBF kernel.
    variance: float
        Variance of the RBF kernel.

    Returns
    -------
    S: np.ndarray
        Smoothing matrix of shape (len(var), len(var)).
    """
    kL = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
    L = np.arange(var.shape[0])
    K = kL.K(L[:, None])
    N = np.diag(var)
    S = np.linalg.solve(K + N, K).T
    return S


class BiasEmulator:
    """Evaluate an AB or DB bias emulator at foreground parameters.

    The first input parameter is always ``Tcal``. Remaining parameters are
    normalized using ``x_transform`` before prediction by the two GPy models.

    Parameters
    ----------
    emu1, emu2 : GPy model
        Trained models for the two bias components.
    names : list[str]
        Ordered input names, beginning with ``"Tcal"``.
    x_transform : tuple[np.ndarray, np.ndarray]
        Mean and scale vectors used to normalize all inputs after ``Tcal``.
    kind : {"AB", "DB"}, default="AB"
        Bias-estimation convention to evaluate.
    S : np.ndarray, optional
        Bin-space smoothing matrix applied to the first DB component.
    N0, N1 : np.ndarray, optional
        Reference-normalized N0 and N1 spectra.
    bins : np.ndarray, optional
        Bin edges associated with the output spectra.
    mvtype : str, optional
        Estimator label used to namespace persisted artifacts.
    """

    __allowed_parameters__ = ["Tcal", "Pcal", "Arad", "Acib", "Atsz", "beta_pol"]

    def __init__(
        self,
        emu1,
        emu2,
        names: list,
        x_transform: tuple,
        kind='AB',
        S=None,
        N0=None,
        N1=None,
        bins=None,
        mvtype=None,
    ):
        self.names = names
        for name in self.names:
            if name not in self.__allowed_parameters__:
                raise ValueError(
                    f"Parameter {name} is not allowed. Allowed parameters are: {self.__allowed_parameters__}"
                )
        self.emu1 = emu1
        self.emu2 = emu2
        self.kind = kind
        self.S = S
        self.N0 = N0
        self.N1 = N1
        self.x_mean, self.x_std = x_transform
        self.bins = bins
        self.mvtype = mvtype

    @staticmethod
    def from_builder(builder, emu1, emu2, kind='AB', S=None):
        """Construct an evaluator from a completed :class:`Builder`.

        Parameters
        ----------
        builder : Builder
            Builder supplying parameter normalization, reference spectrum, and
            bias terms.
        emu1, emu2 : GPy model
            Trained component models to evaluate.
        kind : {"AB", "DB"}, default="AB"
            Bias-estimation convention for the returned emulator.
        S : np.ndarray, optional
            Bin-space smoothing matrix for DB evaluation.

        Returns
        -------
        BiasEmulator
            Evaluator with normalized N0/N1 terms and builder metadata.
        """
        if builder.ref.ndim == 2:
            ref = np.mean(builder.ref, axis=0)
        else:
            ref = builder.ref
        boundaries = builder.boundaries
        x_mean = boundaries.mean(axis=1)
        x_std = np.diff(boundaries, axis=1).flatten()
        return BiasEmulator(
            emu1,
            emu2,
            kind=kind,
            S=S,
            N0=builder.N0 / ref,
            N1=builder.N1 / ref,
            x_transform=(x_mean, x_std),
            bins=builder.bins,
            names=['Tcal'] + builder.names,
            mvtype=builder.mvtype,
        )

    @staticmethod
    def load(dirname, mvtype='MVph'):
        _dirname = startup.Config.path(dirname, mvtype)

        emu1 = GPy.core.model.Model.load_model(startup.Config.path(_dirname, "emu1.zip"))
        emu2 = GPy.core.model.Model.load_model(startup.Config.path(_dirname, "emu2.zip"))
        meta = np.load(startup.Config.path(_dirname, "meta.npz"), allow_pickle=True)

        def fmeta(key):
            value = meta[key]
            if value.shape == () and value.dtype == object:
                value = value.item()
            return value

        assert mvtype == fmeta('mvtype'), f"mvtype mismatch: {mvtype} != {fmeta('mvtype')}"
        return BiasEmulator(
            emu1,
            emu2,
            kind=fmeta('kind'),
            S=fmeta('S'),
            N0=fmeta('N0'),
            N1=fmeta('N1'),
            bins=fmeta('bins'),
            x_transform=(fmeta('x_mean'), fmeta('x_std')),
            names=list(fmeta('names')),
            mvtype=fmeta('mvtype'),
        )

    def dump(self, dirname):
        _dirname = startup.Config.path(dirname, self.mvtype)
        os.makedirs(_dirname, exist_ok=True)
        self.emu1.save_model(startup.Config.path(_dirname, "emu1"))
        self.emu2.save_model(startup.Config.path(_dirname, "emu2"))
        np.savez(
            startup.Config.path(_dirname, "meta.npz"),
            kind=self.kind,
            S=self.S,
            N0=self.N0,
            N1=self.N1,
            bins=self.bins,
            x_mean=self.x_mean,
            x_std=self.x_std,
            names=self.names,
            mvtype=self.mvtype,
        )

    def transX(self, X):
        return (X - self.x_mean) / self.x_std

    def __call__(self, p=None, **kwargs):
        """Evaluate the binned bias spectrum.

        Parameters
        ----------
        p : array-like, optional
            One parameter vector or a batch of vectors in :attr:`names` order.
        **kwargs
            Parameter values keyed by :attr:`names`. Used only when ``p`` is
            omitted.

        Returns
        -------
        np.ndarray
            A one-dimensional spectrum for one input vector, or a two-
            dimensional array with one spectrum per input row.
        """
        if p is None:
            p = np.array([kwargs.pop(k) for k in self.names])
            assert len(kwargs) == 0, f"Unused parameters: {kwargs}"
        p = np.atleast_2d(p)
        Tcal = p[:, 0:1]
        assert p.shape[1] == len(self.names)
        x = self.transX(p[:, 1:])
        y1 = self.emu1.predict(x)[0]
        y2 = self.emu2.predict(x)[0]

        if self.kind == 'AB':
            out = Tcal**4 * (y1 + self.N1) - Tcal**2 * (y2 + self.N0)
            out += self.N0 - self.N1
        elif self.kind == 'DB':
            if self.S is not None:
                y1 = (self.S @ y1.T).T
            out = Tcal**4 * y1 + (Tcal**4 - Tcal**2) * y2
            out += (Tcal**4 - 1) * self.N1 - (Tcal**2 - 1) * self.N0
        else:
            raise ValueError(self.kind)

        if out.shape[0] == 1:
            out = np.squeeze(out, axis=0)
        return out


class Builder:
    """Load simulation spectra and train the component GPy emulators.

    Construction loads and reference-normalizes the supplied spectra, then
    trains the standard-spectrum and RDN0 component models. Use :meth:`make`
    to obtain an AB or DB :class:`BiasEmulator`.
    """

    def __init__(
        self,
        qmc: EmulatorSampler,
        mvtype: str,
        dir_fid: str,
        dir_ref: str,
        rdn0_ref: str,
        rdn0_sample: str,
        Nsamp_std,
        Nsamp_RDN0,
        bins,
        rlz_keys,
        N_N0=99,
        N_N1=99,
    ):
        """
        Build Emulator from raw spectra.

        Parameters
        ----------
        qmc: EmulatorSampler
            EmulatorSampler instance that defines the parameter space and sampling.
        mvtype: str
            Type of the mv estimator, e.g., "MVph"/"PP".
        dir_fid: str
            Directory containing the fiducial simulations (fg=1) for the scaling of the emulator.
        dir_ref: str
            Directory containing the reference simulations (fg=0) for the denominator of the output.
        rdn0_ref: str
            Paths to the reference rdn0.db files. This should inclide `{0}` etc for every key in `rlz_keys`.
        rdn0_sample: str
            Paths to the per-sample rdn0.db files. This should inclide `{0}` etc for every key in `rlz_keys`,
            and additionally include `{{i}}` to indicate the file patterns for each QMC sample. For example:
            "samples/{0}{i}c{1}/rdn0.db" where 0 can be replaced by "AB" and 1 be replaced by cmb_seed 1001
            etc.
        Nsamp_std: int or tuple of int
            Number of 0000 samples to use for the emulator. If a tuple, it should be (Ntotal, Ntrain).
        Nsamp_RDN0: int or tuple of int
            Number of RDN0 samples to use for the emulator. If a tuple, it should be (Ntotal, Ntrain).
        rlz_keys: list of tuple
            List of realization keys to format the file paths
        N_N0/N_N1: int
            Number of N0/N1 samples to use for the emulator. For now these are assumed to be the same for
            fid/ref and sample spectra.
        bins: int
            Number of bins to use for the emulator.
        """
        self.names = qmc.parameters
        self.parameters = qmc.grid
        self.boundaries = qmc.boundaries
        self.mvtype = mvtype
        self.dir_fid = dir_fid
        self.dir_ref = dir_ref
        self.rdn0_ref = rdn0_ref
        self.rdn0_sample = rdn0_sample
        self.rlz_keys = rlz_keys

        self.Nsamp_std = Nsamp_std if isinstance(Nsamp_std, tuple) else (Nsamp_std, Nsamp_std)
        self.Nsamp_RDN0 = Nsamp_RDN0 if isinstance(Nsamp_RDN0, tuple) else (Nsamp_RDN0, Nsamp_RDN0)
        self.bins = bins
        self.seeds_N1 = np.arange(1, N_N1 + 1)
        self.seeds_N0 = np.arange(1, N_N0 + 1)
        # fiducial N0 and N1
        f = startup.Config.path(self.dir_fid, 'cls/n0.db')
        self.N0 = self.load_spec(f, 'n0', mvtype=self.mvtype, seeds=self.seeds_N0)
        f = startup.Config.path(self.dir_fid, 'cls/n1.db')
        self.N1 = self.load_spec(f, 'n1', mvtype=self.mvtype, seeds=self.seeds_N1)

        # load reference and sample, average over AB realization, and normalized.
        self.ref = self.load_ref()
        self.sample, self.sample_RDN0 = self.load_sample()
        self.sample /= self.ref[:, None, :]
        self.sample_RDN0 /= self.ref[:, None, :]

        self.x_std = np.diff(self.boundaries, axis=1).flatten()
        self.x_mean = self.boundaries.mean(axis=1)

        self.emu1, self.emu2 = self.train_emulators()

        # the final objective function and its variance of the mean. These are useful to train the emulator
        # in the DB mode.
        self.D = self.sample[:, : self.Nsamp_RDN0[0]] - self.sample_RDN0
        self.varD = np.mean(np.var(self.D, axis=1), axis=0) / self.sample.shape[0]

    def load_spec(self, path, spec_type, mvtype, seeds):
        assert spec_type.lower() in ['n1', 'rdn0', 'n0']
        table = f'g{mvtype}'
        db = spectrum.ClsDB(startup.Config.path(path), table)
        ops = dict(
            n1='abab+abba-xyxy-xyyx',
            rdn0='xxxx' if isinstance(seeds, int) and seeds == 0 else 'x0x0+x00x+0xx0+0x0x',
            n0='xyxy+xyyx',
        )
        if isinstance(seeds, int) and seeds == 0:
            seeds = [0]
        out = ha.read_sql(seeds=seeds, db=db, ops=ops[spec_type])
        if len(seeds) == 1:
            out = np.squeeze(out, 0)
        else:
            out = np.mean(out, axis=0)
        return ha.bin_spectrum(out, self.bins, verbose=False)[1]

    def _load_ref(self, key):
        """
        Load the reference spectra: 0000-(RDN0-N0) - N1 from the reference (A0) directory.

        Returns
        -------
        spec: np.ndarray
            Debiased (except MCresp) fg=0 spectrum.
        """
        fn0 = startup.Config.path(self.dir_ref, 'cls/n0.db')
        N0 = self.load_spec(fn0, 'n0', mvtype=self.mvtype, seeds=self.seeds_N0)

        frdn0 = startup.Config.path(self.rdn0_ref.format(*key))
        RDN0 = self.load_spec(frdn0, 'rdn0', mvtype=self.mvtype, seeds=self.seeds_N0)
        RDN0 -= N0
        dat = self.load_spec(frdn0, 'rdn0', mvtype=self.mvtype, seeds=0)
        return dat - RDN0 - self.N1

    def _load_sample(self, key):
        """
        Load the sample spectra: 0000-(RDN0-N0) - N1 from the sample directory.

        Returns
        -------
        spec: np.ndarray
            Debiased (except MCresp) fg=1 spectrum.
        """

        def f(i):
            return startup.Config.path(self.rdn0_sample.format(*key), i=i)

        RDN0 = np.array(
            [
                self.load_spec(f(i), 'rdn0', mvtype=self.mvtype, seeds=self.seeds_N0)
                for i in range(self.Nsamp_RDN0[0])
            ]
        )
        dat = np.array(
            [self.load_spec(f(i), 'rdn0', mvtype=self.mvtype, seeds=0) for i in range(self.Nsamp_std[0])]
        )
        return dat - self.N1, RDN0 - self.N0

    def load_ref(self):
        """Load debiased reference spectra for every realization key.

        Returns
        -------
        np.ndarray
            Reference spectra with one row per realization key.
        """
        from joblib import Parallel, delayed

        with Parallel(n_jobs=max(20, len(self.rlz_keys))) as p:
            return np.array(p(delayed(self._load_ref)(x) for x in self.rlz_keys))

    def load_sample(self):
        """Load standard and RDN0 spectra for every realization key.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Standard spectra and RDN0 spectra, each indexed first by
            realization key and then by QMC sample.
        """
        from joblib import Parallel, delayed

        with Parallel(n_jobs=max(20, len(self.rlz_keys))) as p:
            dat, RDN0 = zip(*p(delayed(self._load_sample)(x) for x in self.rlz_keys))
        return np.array(dat), np.array(RDN0)

    def transform(self, X):
        return (X - self.x_mean) / self.x_std

    def train(self, dat, Ntrain, seed=None):
        """Train a multi-output ARD-RBF Gaussian-process emulator.

        Parameters
        ----------
        dat : np.ndarray
            Training targets indexed first by QMC sample.
        Ntrain : int
            Number of QMC samples used for fitting.
        seed : int, optional
            Random seed for selecting the training subset. By default, uses
            the initial samples in Sobol order.

        Returns
        -------
        tuple[GPy.models.GPRegression, np.ndarray]
            Fitted emulator and indices withheld from training.
        """
        Nsamp = dat.shape[0]
        if seed is not None:
            rng = np.random.default_rng(seed)
            seq = rng.permutation(Nsamp)
        else:
            seq = np.arange(Nsamp)
        train_idx = seq[:Ntrain]
        withheld_idx = seq[Ntrain:]
        X_train = self.parameters[train_idx]
        Y_train = dat[train_idx]
        X_train_scaled = self.transform(X_train)
        kernel = GPy.kern.RBF(input_dim=X_train.shape[1], ARD=True)

        emulator = GPy.models.GPRegression(
            X_train_scaled, Y_train, kernel=kernel, noise_var=1e-10, normalizer=None
        )
        # emulator.Gaussian_noise.variance.constrain_bounded(1e-8, 0.1)
        emulator.Gaussian_noise.variance.fix(1e-8)
        emulator.optimize_restarts(num_restarts=10, verbose=False)
        if len(withheld_idx):
            predict, var = emulator.predict(
                self.transform(self.parameters[withheld_idx]), include_likelihood=False
            )
            max_dev = np.max(np.abs(predict - dat[withheld_idx]))
            logger.info(f"maxdev: {max_dev:.3e} over {len(withheld_idx)} validation samples.")
        return emulator, withheld_idx

    def train_emulators(self):
        """Train the standard-spectrum and RDN0 component emulators.

        Returns
        -------
        tuple[GPy.models.GPRegression, GPy.models.GPRegression]
            Emulators trained from the realization-averaged standard and RDN0
            spectra, respectively.
        """
        emu_dat, idx_dat = self.train(np.mean(self.sample, axis=0), Ntrain=self.Nsamp_std[1])
        emu_rdn0, idx_rdn0 = self.train(np.mean(self.sample_RDN0, axis=0), Ntrain=self.Nsamp_RDN0[1])
        return emu_dat, emu_rdn0

    def make(self, kind='DB', lengthscale=None, variance=None):
        """Build an AB or DB evaluator from the trained component models.

        Parameters
        ----------
        kind : {"AB", "DB"}, default="DB"
            Requested bias-estimation convention.
        lengthscale : float, optional
            RBF bin-space smoothing length for DB output. No smoothing is
            applied when omitted.
        variance : float, optional
            RBF variance for smoothing. When omitted with ``lengthscale``, it
            is estimated from the realization-derived DB variance.

        Returns
        -------
        BiasEmulator
            Configured evaluator for the requested convention.
        """
        if kind == 'AB':
            return BiasEmulator.from_builder(self, self.emu1, self.emu2, kind='AB')
        else:
            S = None
            Dmean = np.mean(self.sample, axis=0) - self.emu2.predict(self.transform(self.parameters))[0]
            emu_D, idx_dat = self.train(Dmean, Ntrain=self.Nsamp_std[0])
            if lengthscale is not None:
                if variance is None:
                    var_sample = []
                    for d in Dmean:
                        model = GP1d(d, self.varD, lengthscale=lengthscale)
                        var_sample.append(model.rbf.variance.values[0])
                    variance = np.mean(var_sample)
                    logger.info(f"Estimated RBF variance: {variance:.3e} from {len(var_sample)} samples.")
                S = RBFSmooth(self.varD, lengthscale=lengthscale, variance=variance)
            return BiasEmulator.from_builder(self, emu_D, self.emu2, kind='DB', S=S)

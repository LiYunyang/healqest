"""Write a binned lensing result in Candl's on-disk convention."""

import os
import shutil

import numpy as np


class CobayaWriter:
    """Serialize one Candl lensing dataset directory.

    Arrays use the conventions already used by `healqest.analysis`:
    `bandpower` has shape `(n_bin,)`, `covariance` has shape
    `(n_bin, n_bin)`, and `bpwf` has shape `(n_bin, n_ell)`. Candl
    stores the window function transposed, with the multipole in column zero.

    `n_sims` is the number of independent realizations used for the supplied
    covariance. Candl needs it to apply its Hartlap correction; it cannot be
    recovered from a covariance matrix.

    Extra files required by a selected Candl data model (beams, ILC weights,
    or a CMB data template) can be supplied through `assets`. Its keys are
    paths relative to `output_dir` and its values are existing source files.

    Use `with writer():` when writing a dataset. The output directory is
    created on entry and removed if any write inside the block fails.
    """

    _modes = ("TT", "TE", "EE", "BB", "kk")

    def __init__(
        self,
        output_dir: str,
        *,
        bpwf: np.ndarray,
        n_sims: int,
        dataset_name: str = "SPT3G_wide_KK",
        name_in_yaml: str = "SPT-3G phiphi",
        variant: str = "GMV",
        emulator_path: str | None = None,
        assets: dict[str, str] | None = None,
        mll: dict[str, np.ndarray] | None = None,
        fiducial_correction: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Create a writer for one Candl `kk` likelihood dataset.

        Parameters
        ----------
        output_dir: str
            Destination dataset directory. The writer creates the Candl
            subdirectories it needs below this path.
        bpwf: np.ndarray
            Bandpower window functions with shape `(n_bin, n_ell)`. Row
            `b` maps an unbinned `C_ell^kk` to bandpower `b`. The
            writer transposes this project-native layout and prepends the
            implied `ell = 0, ..., n_ell - 1` column required by Candl.
        n_sims: int
            Number of independent simulations used to estimate `covariance`.
            It is written to Candl's `hartlap_correction.N_sims` field and
            must be greater than `n_bin + 2`.
        dataset_name: str, optional
            Dataset prefix used in output filenames.
        variant: str, optional
            Candl dataset variant used in output filenames, such as `"GMV"`
            or `"PP"`.
        emulator_path: str, optional
            Existing emulator file to copy unchanged to `emulator/`.
        assets: dict[str, str], optional
            Additional Candl data-model files to copy. Keys are paths relative
            to `output_dir`; values are existing source paths. For example,
            `{"beam/B_ell_main_beam.npz": main_beam_path}`.
        mll: dict[str, np.ndarray], optional
            Linear-response matrices for `TT`, `TE`, `EE`, and `kk`. Each
            value has shape `(n_bin, n_ell_mll)`. The writer adds a zero `BB`
            matrix and transposes each selected bin into one Candl correction
            window per bandpower bin. `TT`, `TE`, `EE`, and `BB` multiply
            Candl input `D_ell` spectra in `µK_CMB²`; `kk` multiplies
            `C_ell^kk`. Every matrix therefore produces a correction in
            `C_b^kk`.
        fiducial_correction: dict[str, np.ndarray], optional
            Fiducial response correction for the same `TT`, `TE`, `EE`, and
            `kk` modes as `mll`. Each value has shape `(n_bin,)`. The writer
            adds a zero `BB` vector. Every value is in `C_b^kk` and must be
            supplied together with `mll`.

        Notes
        -----
        Candl applies the linear correction as

        `ΔC_b^kk = Σ_mode (M_mode @ X_mode - F_mode)_b`,

        where `X_mode` is `D_ell` for CMB modes and `C_ell^kk` for the
        lensing mode. `F_mode` is the corresponding fiducial correction.

        Examples
        --------
        >>> writer = CobayaWriter(
        ...     "dataset/GMV", bandpower, bpwf, covariance, n_sims=498,
        ...     dataset_name="SPT3G_D1_KK", variant="GMV",
        ... )
        >>> with writer():
        ...     written = writer.write_all(yaml_template=True)
        """
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.name_in_yaml = name_in_yaml
        self.variant = variant
        self.n_sims = n_sims
        self.bpwf = np.asarray(bpwf, dtype=float)
        self.emulator_path = emulator_path
        self.assets = {} if assets is None else dict(assets)

        if mll is None:
            self.mll = None
            self.fiducial_correction = None
        else:
            self.mll = np.array(
                [mll[mode] if mode != "BB" else np.zeros_like(mll["TT"]) for mode in self._modes], dtype=float
            )
            self.fiducial_correction = np.array(
                [
                    fiducial_correction[mode] if mode != "BB" else np.zeros_like(fiducial_correction["TT"])
                    for mode in self._modes
                ],
                dtype=float,
            )
        self._created_output_dir = False
        self._validate()

    def __call__(self):
        """Return this writer for use as `with writer():`."""
        return self

    def __enter__(self):
        """Create a new output directory and return this writer.

        An existing path is rejected so an interrupted write can never remove
        a dataset that was already present before the context was entered.
        """
        if os.path.exists(self.output_dir):
            raise FileExistsError(f"Output directory already exists: {self.output_dir}")
        os.makedirs(self.output_dir)
        self._created_output_dir = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Remove this context's output directory when the write failed."""
        if exc_type is not None and self._created_output_dir:
            shutil.rmtree(self.output_dir)
            self._created_output_dir = False
        return False

    @property
    def stem(self) -> str:
        return f"{self.dataset_name}_{self.variant}"

    @property
    def n_bin(self) -> int:
        return self.bpwf.shape[0]

    def write_bandpower(self, bandpower: np.ndarray, suffix: str = None) -> str:
        """Write a bandpower file.

        Parameters
        ----------
        bandpower: np.ndarray
        suffix: str, optional
            Label appended to an alternate filename as `<dataset>_<variant>_bdp_<suffix>.txt`.
        """
        target_path = self._add_suffix(f"{self.stem}_bdp.txt", suffix)
        if bandpower is not None:
            bandpower = np.asarray(bandpower, dtype=float)
            if bandpower.shape != (self.n_bin,):
                raise ValueError("alternate bandpower must have shape (n_bin,).")
            self._write_array(target_path, bandpower)
        return target_path

    def write_covariance(self, covariance: np.ndarray, suffix: str = None) -> str:
        """Write a covariance file.

        Parameters
        ----------
        covariance: np.ndarray
        suffix: str, optional
            Label appended to an alternate filename as `<dataset>_<variant>_cov_<tag>_<suffix>.txt`.
        """
        target_path = self._add_suffix(f"{self.stem}_cov.txt", suffix)
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=float)
            assert covariance.shape == (self.n_bin, self.n_bin)
            self._write_array(target_path, covariance)
        return target_path

    def write_windows(self) -> str:
        """Write Candl's ell-by-bin `kk_window_functions.txt`."""
        data = np.column_stack((np.arange(self.bpwf.shape[1]), self.bpwf.T))
        return self._write_array(
            "windows/kk_window_functions.txt", data, fmt=("%d",) + ("%.18e",) * self.n_bin
        )

    def write_mll(self) -> list[str]:
        """Write one six-column linear-correction file for every bandpower bin."""
        self._require_mll()
        paths = []
        for bin_index in range(self.n_bin):
            data = np.column_stack((np.arange(self.mll.shape[-1]), self.mll[:, bin_index, :].T))
            paths.append(
                self._write_array(
                    f"linear_corrections/window_{bin_index}.txt", data, fmt=("%d",) + ("%.18e",) * 5
                )
            )
        return paths

    def write_fiducial_correction(self) -> str:
        """Write the fiducial response in Candl's TT, TE, EE, BB, kk order."""
        self._require_mll()
        return self._write_array("linear_corrections/fiducial_correction.txt", self.fiducial_correction.T)

    def write_emulator(self) -> str | None:
        """Copy the supplied emulator unchanged into `emulator/`."""
        if self.emulator_path is None:
            return None
        outdir = 'emulator/'
        shutil.copytree(self.emulator_path, os.path.join(self.output_dir, outdir))
        return outdir

    def write_assets(self) -> dict[str, str]:
        """Copy additional data-model inputs, preserving their requested paths."""
        return {
            relative_name: self._copy_file(source, relative_name)
            for relative_name, source in self.assets.items()
        }

    def write_yaml_template(self, path_dict, data_model=None, suffix: str = None) -> str:
        """Write a minimal Candl dataset YAML for one bandpower/covariance suffix.

        Use `write_all()` to write this YAML together with the files it
        references. `suffix` selects both alternate data filenames.
        """
        try:
            import yaml
        except ImportError as error:
            raise ImportError("Writing a Candl YAML template requires PyYAML.") from error

        document = {
            "name": self.name_in_yaml,
            "band_power_file": path_dict["bandpower"],
            "covariance_file": path_dict["covariance"],
            "window_functions_folder": "windows/",
            "hartlap_correction": {"N_sims": self.n_sims},
            "spectra_info": [{"kk": self.n_bin}],
        }
        if data_model is not None:
            document["data_model"] = list(filter(lambda x: x, data_model))
        filename = self._add_suffix(f"{self.stem}.yaml", suffix)
        target = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            yaml.safe_dump(document, handle, sort_keys=False)
        return target

    def write_bp_cov(
        self,
        bandpower: np.ndarray = None,
        covariance: np.ndarray = None,
        *,
        yaml_template: bool = True,
        data_model=None,
        suffixs: list = None,
    ) -> dict:
        """Write one internally consistent Candl dataset bundle.

        An alternate dataset requires one shared `suffix`. The supplied
        bandpower and covariance, or their primary values when omitted, are
        written with those suffixes and the YAML references those exact files.
        """
        if isinstance(suffixs, str) or suffixs is None:
            suffixs = [suffixs, suffixs, suffixs]
        written = {
            "bandpower": self.write_bandpower(bandpower, suffix=suffixs[0]),
            "covariance": self.write_covariance(covariance, suffix=suffixs[1]),
        }
        if yaml_template:
            written["yaml"] = self.write_yaml_template(
                path_dict=written, data_model=data_model, suffix=suffixs[2]
            )
        return written

    def write_common(self):
        written = {
            "windows": self.write_windows(),
            "emulator": self.write_emulator(),
            "assets": self.write_assets(),
        }

        if self.mll is not None:
            written["mll"] = self.write_mll()
            written["fiducial_correction"] = self.write_fiducial_correction()
        return written

    def _validate(self) -> None:
        self._validate_arrays()
        self._validate_mll()
        self._validate_assets()

    def _validate_arrays(self) -> None:
        if self.n_bin == 0:
            raise ValueError("bandpower must contain at least one bin.")
        if not isinstance(self.n_sims, (int, np.integer)) or isinstance(self.n_sims, bool):
            raise ValueError("n_sims must be an integer.")
        if self.n_sims <= self.n_bin + 2:
            raise ValueError("n_sims must exceed n_bin + 2 for Candl's Hartlap correction.")
        if self.bpwf.ndim != 2 or self.bpwf.shape[0] != self.n_bin:
            raise ValueError("bpwf must have shape (n_bin, n_ell).")

    def _validate_mll(self) -> None:
        if self.mll is None and self.fiducial_correction is not None:
            raise ValueError("fiducial_correction requires mll.")
        if self.mll is not None:
            self._require_mll()

    def _validate_assets(self) -> None:
        for relative_name in self.assets:
            if os.path.isabs(relative_name) or ".." in os.path.normpath(relative_name).split(os.sep):
                raise ValueError("asset paths must be relative to output_dir.")

    def _require_mll(self) -> None:
        if self.mll is None or self.fiducial_correction is None:
            raise ValueError("mll and fiducial_correction must be supplied together.")
        if self.mll.ndim != 3 or self.mll.shape[0] != len(self._modes) or self.mll.shape[1] != self.n_bin:
            raise ValueError("mll must have shape (5, n_bin, n_ell_mll).")
        if self.fiducial_correction.shape != (len(self._modes), self.n_bin):
            raise ValueError("fiducial_correction must have shape (5, n_bin).")

    def _copy_file(self, source: str, relative_name: str) -> str:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Candl asset does not exist: {source}")
        target = os.path.join(self.output_dir, relative_name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(source, target)
        return target

    @staticmethod
    def _add_suffix(filename: str, suffix: str = None) -> str:
        """Append a suffix before a filename extension."""
        if suffix is None:
            return filename
        stem, extension = os.path.splitext(filename)
        return f"{stem}_{suffix}{extension}"

    def _write_array(self, relative_name: str, data: np.ndarray, fmt="%.18e") -> str:
        target = os.path.join(self.output_dir, relative_name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        np.savetxt(target, data, fmt=fmt)
        return target

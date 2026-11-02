import numpy as np
from getdist import MCSamples
from healqest import log

logger = log.get_logger(__name__)


class Blindfold:
    def __init__(self, seed, **kwargs):
        """
        Initializes the Blindfold with a given mask.

        Parameters
        ----------
        mask : np.ndarray
            A boolean array indicating which parameters are masked (True) and which are not (False).
        """
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.secret = {k: rng.normal(0, v) for k, v in kwargs.items()}

    @property
    def blinded_keys(self):
        return set(self.secret.keys())

    def _apply_dict(self, dataset):
        """
        Applies the blindfolding to a dataset represented as a dictionary.

        Parameters
        ----------
        dataset : dict
            A dictionary where keys are parameter names and values are their corresponding values.

        Returns
        -------
        dict
            A new dictionary with the blindfolding applied.
        """
        unblinded_keys = set(dataset.keys()) - set(self.secret.keys())
        if unblinded_keys:
            logger.warning(
                f"Unblinded keys found in dataset: {unblinded_keys}. They will remain unchanged."
                f"The blinded keys are: {self.secret.keys()}"
            )
        return {k: v + self.secret.get(k, 0) for k, v in dataset.items()}

    def _apply_chain(self, chain):
        blinded = chain.copy()

        names = [p.name for p in blinded.getParamNames().names]
        unblinded_keys = set(names) - self.secret.keys()

        if unblinded_keys:
            logger.warning(
                f"Unblinded keys found in chain: {unblinded_keys}. They will remain unchanged."
                f"The blinded keys are: {self.secret.keys()}"
            )

        for i, name in enumerate(names):
            if name in self.secret:
                blinded.samples[:, i] += self.secret[name]

        blinded.updateBaseStatistics()
        return blinded

    def __call__(self, dataset):
        if isinstance(dataset, dict):
            return self._apply_dict(dataset)
        elif isinstance(dataset, MCSamples):
            return self._apply_chain(dataset)
        else:
            raise NotADirectoryError

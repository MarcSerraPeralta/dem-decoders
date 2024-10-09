from typing import Optional

import numpy as np
import stim
from ldpc.belief_find_decoder import BeliefFindDecoder
from tqdm import tqdm

from .transformations import dem_to_hplc


class BP_UF:
    def __init__(
        self, dem: stim.DetectorErrorModel, verbose: Optional[bool] = None, **kargs_bpuf
    ) -> None:
        """Initialises the BeliefPropagation+UnionFind (BP+UF)

        Parameters
        ----------
        dem
            Decoding graph in the form of ``stim.DetectorErrorModel``.
        verbose
            If specified, the ``verbose`` option in ``decode_batch`` are
            overwritten with the value given here.
        kargs_bpuf
            Dictionary with extra arguments for ``lpdc.belief_find_decoder``.
        """
        h, p, l, _ = dem_to_hplc(dem)
        self.check_matrix = h
        self.priors = p
        self.logical_matrix = l

        # make the default 'uf_method="inversion"' instead of "peeling"
        # so that BP_UF can decode any LDPC code.
        if "uf_method" not in kargs_bpuf:
            kargs_bpuf["uf_method"] = "inversion"

        self._decoder = BeliefFindDecoder(
            self.check_matrix,
            error_channel=self.priors,
            **kargs_bpuf,
        )

        self.verbose = verbose

        return

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decodes a single sample of an experiment.

        Parameters
        ----------
        syndrome: np.ndarray(D)
            Observed (detector) syndrome vector.
            Its length must match ``dem.num_detectors``.

        Returns
        -------
        prediction: np.ndarray(L)
            Prediction of the logical flips in the sample.
            Its length matches ``dem.num_observables``.
        """
        error_mechanisms = self._decoder.decode(syndrome)
        prediction = (self.logical_matrix @ error_mechanisms) % 2
        return prediction

    def decode_to_faults_array(self, syndrome: np.ndarray) -> np.ndarray:
        """Decodes a single sample of an experiment.

        Parameters
        ----------
        syndrome: np.ndarray(D)
            Observed (detector) syndrome vector.
            Its length must match ``dem.num_detectors``.

        Returns
        -------
        np.ndarray(E)
            Prediction of the errors that happened in the sample.
            Its length matches ``dem.num_errors``.
        """
        return self._decoder.decode(syndrome)

    def decode_batch(self, syndromes: np.ndarray, verbose=True) -> np.ndarray:
        """Decodes a several samples of an experiment.

        Parameters
        ----------
        syndromes: np.ndarray(S, D)
            Observed (detector) syndrome vectors for each sample.
            Its shape must be ``(num_shots, dem.num_detectors)``.
        verbose
            If True, prints a progress bar.
            This value is overwritten by the one specified when
            initializing this object.

        Returns
        -------
        predictions: np.ndarray(S, L)
            Prediction of the logical flips for each sample.
            Its shape is ``(num_shots, dem.num_observables)``.
        """
        if self.verbose is not None:
            verbose = self.verbose

        predictions = np.zeros(
            (syndromes.shape[0], self.logical_matrix.shape[0]), dtype=bool
        )
        if verbose:
            for i in tqdm(range(syndromes.shape[0])):
                predictions[i, :] = self.decode(syndromes[i, :])
        else:
            for i in range(syndromes.shape[0]):
                predictions[i, :] = self.decode(syndromes[i, :])
        return predictions

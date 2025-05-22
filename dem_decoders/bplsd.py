from typing import Optional

import numpy as np
import stim
from ldpc.bplsd_decoder import BpLsdDecoder
from tqdm import tqdm

from .transformations import dem_to_hplc


class BP_LSD:
    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        verbose: Optional[bool] = None,
        **kargs_bplsd,
    ) -> None:
        """Initialises the BeliefPropagation+LocalizedStatisticsDecoder (BP+LSD)

        Parameters
        ----------
        dem
            Decoding graph in the form of ``stim.DetectorErrorModel``.
        verbose
            If specified, the ``verbose`` option in ``decode_batch`` are
            overwritten with the value given here.
        kargs_bplsd
            Dictionary with extra arguments for ``lpdc.bplsd_decoder§``.
        """
        h, p, l, _ = dem_to_hplc(dem)
        self.check_matrix = h
        self.priors = p
        self.logical_matrix = l

        self._decoder = BpLsdDecoder(
            self.check_matrix,
            channel_probs=self.priors,
            **kargs_bplsd,
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

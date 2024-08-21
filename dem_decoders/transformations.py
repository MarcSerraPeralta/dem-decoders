from typing import Tuple, List, Optional

import numpy as np
from scipy.sparse import csc_matrix
import stim

from .util import comb_probs


def dem_to_hplc(
    dem: stim.DetectorErrorModel,
) -> Tuple[csc_matrix, np.ndarray, csc_matrix, np.ndarray]:
    """Returns the detector-error matrix, error probabilities, logicals-error matrix,
    and the detector coordinates given a ``stim.DetectorErrorModel``.

    Parameters
    ----------
    dem
        Detector error model (DEM).

    Returns
    -------
    det_err_matrix : np.ndarray(D, E)
        Detector-error matrix which related the error mechanisms and the detectors
        they trigger. ``D`` is the number of detectors and ``E`` the number
        of error mechanisms.
    err_probs : np.ndarray(E)
        Probabilities for each error mechanism.
    log_err_matrix : np.ndarray(L, E)
        Logicals-error matrix which relates the error mechanisms and the logical
        observables that they flip. ``L`` is the number of logical observables.
    coords : np.ndarray(D, C)
        Coordinates associated with each detector, with ``C`` the number of coordinates.
        If no coordinates are present in ``dem``, an empty array of shape ``(D,)``
        is returned.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )

    det_err_list = []
    err_probs_list = []
    log_err_list = []
    coords_dict = {}

    for instr in dem.flattened():
        if instr.type == "error":
            # get information
            p = instr.args_copy()[0]
            dets, logs = [], []
            for t in instr.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    logs.append(t.val)
                elif t.is_separator():
                    pass
                else:
                    raise ValueError(f"{t} is not implemented.")
            # append information
            if dets in det_err_list:
                idx = det_err_list.index(dets)
                if set(logs) != set(log_err_list[idx]):
                    raise ValueError(
                        f"Error {dets} and {det_err_list[idx]} trigger the same detectors,"
                        " but have different logical effect."
                    )
                err_probs_list[idx] = comb_probs(p, err_probs_list[idx])
            else:
                det_err_list.append(dets)
                err_probs_list.append(p)
                log_err_list.append(logs)
        elif instr.type == "detector":
            det = instr.targets_copy()[0].val
            coords_dict[det] = instr.args_copy()
        elif instr.type == "logical_observable":
            pass
        else:
            raise ValueError(f"{instr} is not implemented.")

    det_err_matrix = _list_to_csc_matrix(
        det_err_list, shape=(dem.num_detectors, len(det_err_list))
    )
    log_err_matrix = _list_to_csc_matrix(
        log_err_list, shape=(dem.num_observables, len(log_err_list))
    )
    err_probs = np.array(err_probs_list)
    coords = np.empty(shape=(dem.num_detectors))
    if coords_dict:
        if dem.num_detectors != len(coords_dict):
            raise ValueError(
                "Either all the detectors have coordinates or none,"
                " but not all of them have."
            )
        coords = np.array([coords_dict[i] for i in range(dem.num_detectors)])

    return det_err_matrix, err_probs, log_err_matrix, coords


def hplc_to_dem(
    det_err_matrix: csc_matrix | np.ndarray,
    err_probs: np.ndarray | List[float],
    log_err_matrix: csc_matrix | np.ndarray,
    coords: Optional[np.ndarray] = None,
) -> stim.DetectorErrorModel:
    """Returns the corresponding ``stim.DetectorErrorModel`` from the specified
    detector-error matrix, error probabilities, logicals-error matrix and coordinates.

    Parameters
    ----------
    det_err_matrix : np.ndarray(D, E)
        Detector-error matrix which related the error mechanisms and the detectors
        they trigger. ``D`` is the number of detectors and ``E`` the number
        of error mechanisms.
    err_probs : np.ndarray(E)
        Probabilities for each error mechanism.
    log_err_matrix : np.ndarray(L, E)
        Logicals-error matrix which relates the error mechanisms and the logical
        observables that they flip. ``L`` is the number of logical observables.
    coords : np.ndarray(D, ...)
        Coordinates associated with each detector.
        If no coordinates are present in ``dem``, an empty array of shape ``(D,)``
        is returned.

    Returns
    -------
    dem
        Detector error model.
    """
    if not (
        isinstance(det_err_matrix, csc_matrix) or isinstance(det_err_matrix, np.ndarray)
    ):
        raise TypeError(
            "'det_err_matrix' must be a 'csc_matrix' or a 'ndarray',"
            f" but {type(det_err_matrix)} was given."
        )
    if not (
        isinstance(log_err_matrix, csc_matrix) or isinstance(log_err_matrix, np.ndarray)
    ):
        raise TypeError(
            "'log_err_matrix' must be a 'csc_matrix' or a 'ndarray',"
            f" but {type(log_err_matrix)} was given."
        )
    if not (isinstance(err_probs, list) or isinstance(err_probs, np.ndarray)):
        raise TypeError(
            "'err_probs' must be a 'list' or a 'ndarray',"
            f" but {type(err_probs)} was given."
        )
    if (
        det_err_matrix.shape[1] != log_err_matrix.shape[1]
        or len(err_probs) != det_err_matrix.shape[1]
    ):
        raise ValueError("All inputs must have the same number of error mechanisms.")

    dem = stim.DetectorErrorModel()
    for p, dets, logs in zip(err_probs, det_err_matrix.T, log_err_matrix.T):
        det_targets = [stim.target_relative_detector_id(d) for d in dets.nonzero()[0]]
        log_targets = [stim.target_logical_observable_id(l) for l in logs.nonzero()[0]]
        dem.append(stim.DemInstruction("error", [p], det_targets + log_targets))

    if coords is not None:
        for d, coord in enumerate(coords):
            d_target = stim.target_relative_detector_id(d)
            dem.append(stim.DemInstruction("detector", coord, [d_target]))

    return dem


def _list_to_csc_matrix(my_list: List[List[int]], shape: Tuple[int, int]) -> csc_matrix:
    """Returns ``csc_matrix`` built form the given list.

    The output matrix has all elements zero except in each column ``i`` it has
    ones on the rows ``my_list[i]``.

    Parameters
    ----------
    my_list
        List of lists of integers containing the entries with ones in the csc_matrix.
    shape
        Shape of the ``csc_matrix``.

    Returns
    -------
    matrix
        The described ``csc_matrix`` with 0s and 1s.
    """
    if shape[1] < len(my_list):
        raise ValueError(
            "The shape of the csc_matrix is not large enough to accomodate all the data."
        )

    num_ones = sum(len(l) for l in my_list)
    data = np.ones(
        num_ones, dtype=np.uint8
    )  # smallest integer size (bool operations do not work)
    row_inds = np.empty(num_ones, dtype=int)
    col_inds = np.empty(num_ones, dtype=int)
    i = 0
    for c, det_inds in enumerate(my_list):
        for r in det_inds:
            row_inds[i] = r
            col_inds[i] = c
            i += 1

    return csc_matrix((data, (row_inds, col_inds)), shape=shape)

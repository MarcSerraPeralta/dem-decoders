import numbers
import stim


def comb_two_probs(p1: float, p2: float) -> float:
    """Returns the probability that only one of the events happens.

    Parameters
    ----------
    p1
        Probability of the first event.
    p2
        Probability of the second event.

    Returns
    -------
    float
        Probability that only one of the events happens.
    """
    if (p1 < 0) or (p1 > 1) or (p2 < 0) or (p2 > 1):
        raise ValueError(f"p1={p1} and p2={p2} must be probabilities.")
    return p1 * (1 - p2) + (1 - p1) * p2


def comb_probs(*probs) -> float | int:
    """Returns the probability that an odd number of events happens.

    Parameters
    ----------
    probs
        Iterable of probabilities.
        It can also be a tuple of a single iterable of probabilities.

    Returns
    -------
    odd_prob
        Probability that an off number of events happens.
    """
    # if a list or a numpy.array is passed as input
    if len(probs) == 1 and (not isinstance(probs[0], numbers.Number)):
        probs = probs[0]
    if len(probs) == 0:
        return 0

    odd_prob = probs[0]
    for p in probs[1:]:
        odd_prob = comb_two_probs(odd_prob, p)

    return odd_prob


def remove_gauge_detectors(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(f"'dem' is not a stim.DetectorErrorModel, but a {type(dem)}.")

    new_dem = stim.DetectorErrorModel()
    
    # first scan for all gauge detectors because they don't 
    # have to appear at the beggining of the dem.
    gauge_detectors = []
    for instr in dem.flattened():
        if instr.type == "error" and instr.args_copy()[0] == 0.5:
            gauge_detectors += instr.targets_copy() 
    gauge_detectors = list(set(gauge_detectors)) # get unique gauge detectors
    gauge_detectors_val = sorted([d.val for d in gauge_detectors])

    def shift(target):
        """If ``target`` is a detector, it shifts its index to remove
        the gauge detectors."""
        if not target.is_relative_detector_id():
            return target
            
        val = target.val
        for shift, g_val in enumerate(gauge_detectors_val):
            if val <= g_val:
                return stim.target_relative_detector_id(val - shift)
        return stim.target_relative_detector_id(val - shift - 1)

    def valid_targets(targets):
        dets = [t for t in targets if t.is_relative_detector_id()]
        logs = [t for t in targets if t.is_logical_observable_id()]
        if len(dets) == 0 and len(logs) != 0:
            raise ValueError("An error mechanism does not trigger any errors" 
                             f" but leads to a logical error: {targets}")
        if len(dets) == 0:
            return False
        return True
            
    # remove the gauge detectors 
    for instr in dem.flattened():
        if instr.type == "error":
            new_targets = [t for t in instr.targets_copy() if t not in gauge_detectors]
            if valid_targets(new_targets):
                # remove separators at the beginning or end of new_targets
                # because it leads to an error when creating the 'stim.DemInstruction'.
                # it must be checked after the targets are valid so that 'new_targets'
                # is not empty. 
                if new_targets[0].is_separator():
                    new_targets = new_targets[1:]
                if new_targets[-1].is_separator():
                    new_targets = new_targets[:-1]
                # reindex the detectors
                new_targets = [shift(t) for t in new_targets]
                
                new_dem.append(stim.DemInstruction("error", args=instr.args_copy(), targets=new_targets))

    return new_dem

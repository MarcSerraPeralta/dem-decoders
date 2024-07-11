def comb_probs(p1: float, p2: float):
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

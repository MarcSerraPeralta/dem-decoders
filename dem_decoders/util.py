import numbers


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

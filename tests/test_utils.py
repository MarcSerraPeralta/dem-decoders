import dem_utils.utils as utils


def test_comb_probs():
    assert utils.comb_probs(0.1, 0.2) == 0.26
    return

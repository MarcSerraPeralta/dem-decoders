import dem_decoders.util as util


def test_comb_probs():
    assert util.comb_probs(0.1, 0.2, 0.3) == 0.404
    return

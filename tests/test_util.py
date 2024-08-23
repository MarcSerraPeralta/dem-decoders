import stim

import dem_decoders.util as util


def test_comb_probs():
    assert util.comb_probs(0.1, 0.2, 0.3) == 0.404
    return


def test_remove_gauge_detectors():
    dem = stim.DetectorErrorModel(
        """
        error(0.5) D0
        error(0.1) D1
                              """
    )
    expected_dem = stim.DetectorErrorModel("error(0.1) D1")

    new_dem = util.remove_gauge_detectors(dem)

    assert new_dem == expected_dem

    return

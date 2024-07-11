import numpy as np
import stim

import dem_utils.transformations as transformations
from dem_utils import dem_to_hplc, hplc_to_dem


def test_dem_to_hplc():
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        rounds=5,
        distance=3,
        before_round_data_depolarization=0.1,
    )
    dem = circuit.detector_error_model()

    h, p, l, c = dem_to_hplc(dem)

    assert h.shape == (6 * 2, 5 * 3)
    assert l.shape == (1, 5 * 3)
    assert p.shape == (5 * 3,)
    assert c.shape == (6 * 2,)

    return


def test_hplc_to_dem():
    h = np.array([[1, 1, 0], [0, 1, 1]])
    p = [0.1, 0.2, 0.3]
    l = np.array([[1, 1, 0], [0, 1, 0]])

    dem = hplc_to_dem(h, p, l)

    correct_dem = stim.DetectorErrorModel(
        """
                            error(0.1) D0 L0
                            error(0.2) D0 D1 L0 L1
                            error(0.3) D1
                            """
    )

    assert dem == correct_dem

    return

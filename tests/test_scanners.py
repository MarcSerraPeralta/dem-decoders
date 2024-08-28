import stim

from dem_decoders import scanners


def test_bposd_scanner():
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()

    output = scanners.bposd_scanner(dem, [1, 2], [0, 1], 1)

    assert set(output.keys()) == set([(1, 0), (1, 1), (2, 0), (2, 1)])
    for val in output.values():
        assert len(val) == 2
        assert val[0] >= 0 and val[1] >= 0

    return

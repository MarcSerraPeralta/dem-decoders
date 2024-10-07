import numpy as np
import stim
from dem_decoders import BP_OSD


def test_bp_osd():
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=1e-3,
        before_measure_flip_probability=1e-3,
    )
    sampler = circuit.compile_detector_sampler()
    detectors, log_flips = sampler.sample(shots=1_000, separate_observables=True)

    dem = circuit.detector_error_model()
    bposd = BP_OSD(dem)

    predictions = bposd.decode_batch(detectors)
    assert predictions.shape == (1_000, 1)
    assert np.average(predictions != log_flips) < 0.1

    output = bposd.decode(detectors[0])
    assert output.shape == (dem.num_observables,)

    output = bposd.decode_to_faults_array(detectors[0])
    assert output.shape == (dem.num_errors,)

    return

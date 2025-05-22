from typing import List, Dict, Tuple
import time

import numpy as np
import stim
import matplotlib.pyplot as plt

from .bposd import BP_OSD


def bposd_scanner(
    dem: stim.DetectorErrorModel,
    max_iters: List[int],
    osd_orders: List[int],
    max_time_scan: float,
    num_max_fails: int | float = np.inf,
    num_batch_samples: int = 1_000_000,
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Scans the ``max_iter`` and ``osd_order`` parameters of ``BP_OSD``
    given the specified grid.

    Parameters
    ----------
    dem
        Detector error model or decoding graph.
    max_iters
        List of maximum number of BP iterations to scan.
    osd_orders
        List of OSD orders to scan.
        Note that the ``CS`` method is used.
    max_time_scan
        Maximum time to spend in this scan (in seconds).
    num_max_fails
        Maximum number of decoding failures to compute, after that the
        computation for the given pair of parameters is stopped.
    num_batch_samples
        Number of samples to simulate from the ``dem`` in every batch
        for the decoder to decode.

    Returns
    -------
    output
        Dictionary containing the logical error probability and the average
        decoder runtime per shot (in seconds) for each pair of
        ``(max_iter, osd_order)``.

    Notes
    -----
    First, it is recommended to scan the decoder speed by setting a very low
    ``max_time_scan`` to check the parameter regime that best suits the
    amount of time one has for decoding.
    Secondly, in the selected regime, run another scan with
    ``num_max_fails = 2`` or ``5`` to have a quick (though unprecise) look at
    the logical performance.

    If the physical error rate is high (:math:`\\sim 10^{-2}`), set the
    ``num_max_fails = np.inf`` because the logical error probability is
    close to :math:`1/2`.
    """
    t_init_estimation = time.time()
    num_points = len(max_iters) * len(osd_orders)
    output = {}

    for max_iter in max_iters:
        for osd_order in osd_orders:
            print(f"{max_iter=} {osd_order=}")
            t_init_point = time.time()

            decoder = BP_OSD(
                dem, max_iter=max_iter, osd_order=osd_order, osd_method="OSD_CS"
            )

            # compute available time
            elapsed_time = t_init_point - t_init_estimation
            remaining_points = num_points - len(output)
            max_duration = (max_time_scan - elapsed_time) / remaining_points

            # sample data
            sampler = dem.compile_sampler()
            detectors, log_flips, _ = sampler.sample(shots=num_batch_samples)

            # decode data
            runtime_decoder, num_total, num_fail, idx = 0, 0, 0, 0
            latest_print_time = time.time()
            while (
                time.time() - t_init_point
            ) < max_duration and num_fail < num_max_fails:
                # generate new data (if needed)
                if len(detectors) == num_total:
                    idx = 0
                    detectors, log_flips, _ = sampler.sample(shots=num_batch_samples)

                detector_vec, log_flip = detectors[idx], log_flips[idx]

                t0 = time.time()
                prediction = decoder.decode(detector_vec)
                runtime = time.time() - t0

                # update variables
                num_total += 1
                num_fail += (prediction != log_flip).any()
                runtime_decoder += runtime

                if time.time() - latest_print_time > 1:  # update every second
                    latest_print_time = time.time()
                    print(
                        f"\r    p={num_fail/num_total:0.9f}",
                        "t={np.average(runtime_decoder / num_total):0.6f}s",
                        "num_fail={num_fail}",
                        "num_total={num_total}",
                        end="",
                    )
            # keep latest version on screen
            print(
                f"\r    p={num_fail/num_total:0.9f}",
                "t={np.average(runtime_decoder / num_total):0.6f}s",
                "num_fail={num_fail}",
                "num_total={num_total}",
            )

            # store data
            output[(max_iter, osd_order)] = [
                num_fail / num_total,
                runtime_decoder / num_total,
            ]

    return output


def plot(
    data: Dict[Tuple[int, int], Tuple[float, float]],
    num_min_fails: int = 100,
    cmap=plt.cm.Blues,
) -> plt.Figure:
    """Returns a figure containing a summary of the given scan.

    The figure has three subplots corresponding to:
    (1) logical error rate probability,
    (2) decoder runtime per sample,
    (3) required time to decode the data and observe ``num_min_fails``
    decoding failures.

    Parameters
    ----------
    data
        Output of the scanners.
    num_min_fails
        Minimum number of fails to observe when decoding.
    cmap
        ``matplotlib`` colormap.

    Returns
    -------
    fig
        ``matplotlib`` figure.
    """
    # unique max_iter and order
    max_iters = sorted(list(set([i[0] for i in data.keys()])))
    orders = sorted(list(set([i[1] for i in data.keys()])))

    # compute matrices
    log_err_prob = np.zeros((len(max_iters), len(orders)))
    runtime_decoder = np.zeros((len(max_iters), len(orders)))
    runtime_experiment = np.zeros((len(max_iters), len(orders)))

    for key in data:
        max_iter_idx, order_idx = max_iters.index(key[0]), orders.index(key[1])
        prob, runtime = data[key]

        log_err_prob[max_iter_idx, order_idx] = prob
        runtime_decoder[max_iter_idx, order_idx] = runtime
        runtime_experiment[max_iter_idx, order_idx] = (
            num_min_fails / prob * runtime if prob != 0 else np.nan
        )

    log_err_prob = np.log10(log_err_prob)
    runtime_decoder = np.log10(runtime_decoder)
    runtime_experiment = runtime_experiment / 3600

    # plot the matrices
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for ax_idx, matrix in enumerate(
        [log_err_prob, runtime_decoder, runtime_experiment]
    ):
        axes[ax_idx].imshow(matrix, cmap=cmap)
        for i, vec in enumerate(matrix.T):
            for j, value in enumerate(vec):
                axes[ax_idx].text(i, j, f"{value:0.2f}", va="center", ha="center")
        axes[ax_idx].set_xticks(range(len(orders)), map(str, orders))
        axes[ax_idx].set_yticks(range(len(max_iters)), map(str, max_iters))
        axes[ax_idx].set_xlabel("order")
    axes[0].set_ylabel("max_bp_iter")

    axes[0].set_title("$\\log_{10}(p_L)$")
    axes[1].set_title("$\\log_{10}($decoding time [s]/shot$)$")
    axes[2].set_title("required time for $n_{fail} = " + f"{num_min_fails}" + "$ [h]")

    return fig

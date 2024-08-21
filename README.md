# dem-utils

![example workflow](https://github.com/MarcSerraPeralta/dem-decoders/actions/workflows/actions.yaml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI](https://img.shields.io/pypi/v/dem-decoders?label=pypi%20package)

Wrapped decoders to work with `stim.DetectorErrorModel`.

The methods that allow to transform between (parity-check) matrices & priors and `stim.DetectorErrorModel`s are:

1. `dem_decoders.transformations.dem_to_hplc`
1. `dem_decoders.transformations.hplc_to_dem`

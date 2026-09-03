# H2MMbursts

[![Tests](https://github.com/OpenSMFS/H2MMbursts/actions/workflows/test.yml/badge.svg)](https://github.com/OpenSMFS/H2MMbursts/actions)
[![Documentations Status](https://readthedocs.org/projects/H2MMbursts/badge?version=latest)](https://h2mmbursts.readthedocs.io/en/latest/?badge=latest)


## Project Description

Extension of smfBursts for working with H<sup>2</sup>MM.
H2MMbursts provides the `StatePath` and `Dwell` tables which represent the results of *Viterbi* processing burst data, finding the most likely state of each photon for a given H<sup>2</sup>MM model.

## Install

Install with 

```bash
pip install H2MMbursts
```

## Features

### *Viterbi* based classes

`H2MMbursts` adds a new Table classes for using H<sup>2</sup>MM with `smfBursts`
The two primary table types are `StatePath` and `Dwells`.

`StatePath` records (per burst) the state of each photon (it is a `ChildPhotonTable`).
Use the `StatePath.optimize_models()` classmethod to perform H<sup>2</sup>MM optimizations
on a given set of data, this will create a set of `StatePath` based on the optimized models
of increasing numbers of states.

While `Dwells` is a `BasePhotonTable` that separates bursts into dwells of consecutive photons
of the same state, allowing the user to treat these dwells like bursts.

### Error analysis

The function `bhm.error.statepath_ll_error()` is used to assess the error of each parameter
for a given H<sup>2</sup>MM model based on the data.

### Simulations

The `bhm.sim.H2MMSim` table also allows a Monte-Carlo recoloring of photons in data,
which can be used to compare the predicted results of a model vs the actual.
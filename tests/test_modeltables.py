# Created: 06/07/2026
# author: Paul David harris
"""
Testing the modeltables module of burstH2MM
"""
from itertools import chain

import numpy as np

import smfbursts as smf
import burstH2MM as bhm

import pytest


def test_statepath(hp3, streams, brst):
    srt = bhm.StatePath.sort_photons(hp3, bursts=brst)
    stream = streams[0]
    for s in streams:
        stream |= s
    times = hp3.get_column(smf.Column(brst, 'ph_times', stream))
    assert srt['times'].shape == times.shape
    for ti, t in zip(srt['times'], times):
        assert ti.shape == t.shape
    

def test_statepath_optimize(hp3, brst, streams):
    path = bhm.StatePath.optimize(hp3, brst, bhm.factory_h2mm_model(2, len(streams)), 
                                  streams=streams)
    assert isinstance(path, smf.Param), 'bhm.StatePath.optimize, wrong return type'
    assert path.tp == bhm.StatePath, 'bhm.StatePath.optimize, wrong Param type'
    assert path.params['model'].nstate == 2
    assert path.params['model'].ndet == len(streams)
    assert len(path.params['streams']) == len(streams)
    assert all(stream in streams for stream in path.params['streams'])
    ball = brst.origin_param
    pathg = bhm.StatePath.optimize(hp3, ball, bhm.factory_h2mm_model(2, len(streams)), 
                                   streams=streams, gate=brst.base_gate)
    assert np.allclose(path.params['model'].prior, pathg.params['model'].prior)
    assert np.allclose(path.params['model'].trans, pathg.params['model'].trans)
    assert np.allclose(path.params['model'].obs, pathg.params['model'].obs)


def dicts_merge(*dcts):
    return {k:v for k, v in chain.from_iterable(dct.items() for dct in dcts)}
    

ccrits = ['BICph', 'BIC', 'BICp','ICL', 'ICLph', 'pathBIC', 'pathBICph']
opt_kw_base = {'min_states':2,  'to_state':3, 'max_state':4}
opt_kws = [dicts_merge(opt_kw_base, {'conv_crit':ccrit}) for ccrit in ccrits]
for opt_kw in opt_kws:
    if opt_kw['conv_crit'].endswith('ph') or opt_kw['conv_crit'].endswith('p'):
        opt_kw['thresh'] = 0.005
    

@pytest.fixture(params=opt_kws)
def sp_opt_kwargs(request):
    return request.param


def test_optimize_models(hp3, brst, streams, sp_opt_kwargs):
    paths = bhm.StatePath.optimize_models(hp3, brst, **sp_opt_kwargs)
    assert len(paths) > 1 and len(paths) < 4, 'number of outputs out of range'
    nstates = np.array([p.params['model'].nstate for p in paths])
    assert np.all(np.diff(nstates == 1)), 'Skipping/out of order state in optimize models'
    assert nstates[0] == 2


pathcolumns = (('timepath', np.int64, 1), ('detpath', np.uint8, 1), ('indexpath', np.uint8, 1), 
               ('statepath', np.uint8, 1), ('scalepath', np.float64, 1), 
               ('llpath', np.float64, 1), ('gammapath', np.float64, 2))

@pytest.fixture(params=pathcolumns)
def pathcolumn_types(request):
    return request.param


def test_pathcolumns(data, statepath, pathcolumn_types):
    col, dtype, nd = pathcolumn_types
    carray = data.get_column(smf.Column(statepath, col))
    sort = bhm.StatePath.sort_photons(data, statepath)
    for carr, idx in zip(carray, sort['indexes']):
        assert isinstance(carr, np.ndarray)
        assert np.issubdtype(carr.dtype, dtype)
        assert carr.ndim == nd
        assert carr.shape[0] == idx.shape[0]
        if nd > 1:
            assert carr.shape[1] == statepath.params['model'].nstate

# Created: 06/07/2026
# author: Paul David harris
"""
Testing the modeltables module of burstH2MM
"""
from itertools import chain, product

import numpy as np

import smfbursts as smf
import H2MMbursts as bhm

import pytest


bhm.optimization_limits.squarem = True

def test_statepath(hp3, streams, brst):
    srt = bhm.StatePath.sort_photons(hp3, bursts=brst, streams=streams)
    stream = streams[0]
    for s in streams[1:]:
        stream |= s
    times = hp3.get_column(smf.Column(brst, 'ph_times', stream))
    assert srt['times'].shape == times.shape
    for i, (ti, t) in enumerate(zip(srt['times'], times)):
        assert ti.shape == t.shape, f'Mismatched shape {ti.shape} vs {t.shape} at burst {i}'

    
def test_ntstatepath(hp3, streams, brst):
    divs = [np.array([2500, 2600]), np.array([2600,]), np.array([300])][:len(streams)]
    ndiv = sum(arr.size+1 for arr in divs)
    srt = bhm.ntdivStatePath.sort_photons(hp3, bursts=brst, streams=streams, divs=divs)
    stream = streams[0]
    for s in streams[1:]:
        stream |= s
    times = hp3.get_column(smf.Column(brst, 'ph_times', stream))
    assert srt['times'].shape == times.shape
    for i, (ti, t) in enumerate(zip(srt['times'], times)):
        assert ti.shape == t.shape, f'Mismatched shape {ti.shape} vs {t.shape} at burst {i}'
    assert np.all(np.unique(np.concatenate(srt['indexes'])) == np.arange(ndiv))


def test_usalexstatepath(yopo, brst, usalexshifts):
    streams = (smf.PhSel('0ex0em'), smf.PhSel('0ex1em'), smf.PhSel('1ex1em'))
    srt = bhm.usAlexStatePath.sort_photons(yopo, bursts=brst, streams=streams, shifts=usalexshifts)
    stream = streams[0]
    for s in streams[1:]:
        stream |= s
    times = yopo.concatenate_column(smf.Column(brst, 'ph_times', smf.PhSel('0ex_1ex1em')))
    assert srt['times'].shape == times.shape
    for i, (ti, t) in enumerate(zip(srt['times'], times)):
        assert ti.shape == t.shape, f'Mismatched shape {ti.shape} vs {t.shape} at burst {i}'
    assert np.all(np.unique(np.concatenate(srt['indexes'])) == np.arange(3))


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


def test_ntstatepath_optimize(hp3, brst, streams):
    divs = [np.array([2500, 2600]), np.array([2600,]), np.array([300])][:len(streams)]
    ndiv = sum(div.size+1 for div in divs)
    path = bhm.ntdivStatePath.optimize(hp3, brst, bhm.factory_h2mm_model(2, ndiv), 
                                       streams=streams, 
                                       divs=divs)
    assert isinstance(path, smf.Param), 'bhm.nddivStatePath.optimize, wrong return type'
    assert path.tp == bhm.ntdivStatePath, 'bhm.ntdivStatePath.optimize, wrong Param type'
    assert path.params['model'].nstate == 2
    assert path.params['model'].ndet == ndiv
    assert len(path.params['streams']) == len(streams)
    assert all(stream in streams for stream in path.params['streams'])
    ball = brst.origin_param
    pathg = bhm.ntdivStatePath.optimize(hp3, ball, bhm.factory_h2mm_model(2, ndiv), 
                                        streams=streams, gate=brst.base_gate, divs=divs)
    assert np.allclose(path.params['model'].prior, pathg.params['model'].prior)
    assert np.allclose(path.params['model'].trans, pathg.params['model'].trans)
    assert np.allclose(path.params['model'].obs, pathg.params['model'].obs)


def test_usstatepath_optimize(yopo, brst):
    streams = (smf.PhSel('0ex0em'), smf.PhSel('0ex1em'), smf.PhSel('1ex1em'))
    shifts = ('base', 'base', 'neven:0')
    path = bhm.usAlexStatePath.optimize(yopo, brst, bhm.factory_h2mm_model(2, 3), 
                                       streams=streams, shifts=shifts)
    assert isinstance(path, smf.Param), 'bhm.usAlexStatePath.optimize, wrong return type'
    assert path.tp == bhm.usAlexStatePath, 'bhm.usAlexStatePath.optimize, wrong Param type'
    assert path.params['model'].nstate == 2
    assert path.params['model'].ndet == 3
    assert len(path.params['streams']) == len(streams)
    assert all(stream in streams for stream in path.params['streams'])
    ball = brst.origin_param
    pathg = bhm.usAlexStatePath.optimize(yopo, ball, bhm.factory_h2mm_model(2, len(streams)), 
                                         streams=streams, gate=brst.base_gate, shifts=shifts)
    assert np.allclose(path.params['model'].prior, pathg.params['model'].prior)
    assert np.allclose(path.params['model'].trans, pathg.params['model'].trans)
    assert np.allclose(path.params['model'].obs, pathg.params['model'].obs)


def dicts_merge(*dcts):
    return {k:v for k, v in chain.from_iterable(dct.items() for dct in dcts)}
    

ccrits = ['BICph', 'BIC', 'BICp','ICL', 'ICLph', 'pathBIC', 'pathBICph']
opt_kw_base = {'min_states':2,  'to_state':3, 'max_states':4}
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
    assert np.all(np.diff(nstates) == 1), 'Skipping/out of order state in optimize models, '
    assert nstates[0] == 2


pathcolumns = (('timepath', np.int64, 1), ('detpath', np.uint8, 1), ('indexpath', np.uint8, 1), 
               ('statepath', np.uint8, 1), ('scalepath', np.float64, 1), 
               ('pathllpath', np.float64, 1), ('gammapath', np.float64, 2))

@pytest.fixture(params=pathcolumns)
def pathcolumn_types(request):
    return request.param


def test_pathcolumns(hp3, optsp, pathcolumn_types):
    col, dtype, nd = pathcolumn_types
    carray = hp3.get_column(smf.Column(optsp, col))
    sort = bhm.StatePath.sort_photons(hp3, optsp)
    for carr, idx in zip(carray, sort['indexes']):
        assert isinstance(carr, np.ndarray)
        assert np.issubdtype(carr.dtype, dtype)
        assert carr.ndim == nd
        assert carr.shape[0] == idx.shape[0]
        if nd > 1:
            assert carr.shape[1] == optsp.params['model'].nstate


def test_ndstatepath_columns(hp3, ntstatepath, pathcolumn_types):
    col, dtype, nd = pathcolumn_types
    carray = hp3.get_column(smf.Column(ntstatepath, col))
    sort = bhm.ntdivStatePath.sort_photons(hp3, ntstatepath)
    for carr, idx in zip(carray, sort['indexes']):
        assert isinstance(carr, np.ndarray)
        assert np.issubdtype(carr.dtype, dtype)
        assert carr.ndim == nd
        assert carr.shape[0] == idx.shape[0]
        if nd > 1:
            assert carr.shape[1] == ntstatepath.params['model'].nstate


def test_usalexstatepath_columns(yopo, usstatepath, pathcolumn_types):
    col, dtype, nd = pathcolumn_types
    carray = yopo.concatenate_column(smf.Column(usstatepath, col))
    sort = bhm.usAlexStatePath.sort_photons(yopo, usstatepath)
    for carr, idx in zip(carray, sort['indexes']):
        assert isinstance(carr, np.ndarray)
        assert np.issubdtype(carr.dtype, dtype)
        assert carr.ndim == nd
        assert carr.shape[0] == idx.shape[0]
        if nd > 1:
            assert carr.shape[1] == usstatepath.params['model'].nstate


def test_statepathfilter(hp3, optsp):
    filt = smf.Param(bhm.StatePathFilter, statepath=optsp, exclude={0, 1})
    hp3.get_table(filt)['nph_raw', smf.PhSel('0ex0em')]
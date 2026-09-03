from itertools import product

import numpy as np

import smfbursts as smf
import H2MMbursts as bhm

import pytest


bhm.optimization_limits.squarem = True

def test_bootstrap(hp3, optsp):
    err = bhm.error.BootStrapError.evaluate(hp3, optsp, n=10)
    # smoke tests
    assert err.n == 10
    assert err.model == optsp.model
    assert err.std_prior.shape == optsp.model.prior.shape
    assert err.std_trans.shape == optsp.model.trans.shape
    assert err.std_obs.shape == optsp.model.obs.shape
    assert err.std_prior.shape == optsp.model.prior.shape
    assert err.std_trans.shape == optsp.model.trans.shape
    assert err.std_obs.shape == optsp.model.obs.shape
    assert np.allclose(err.std_prior/np.sqrt(10), err.err_prior)
    assert np.allclose(err.std_trans/np.sqrt(10), err.err_trans)
    assert np.allclose(err.std_obs/np.sqrt(10), err.err_obs)
    E = smf.Column(optsp.bursts, 'E_raw')
    e_std = err.col_std(E)
    e_err = err.col_error(E)
    e_model = np.array([m.obs[:,1]/m.obs[:,:2].sum(axis=1) for m in err.models])
    e_std_expect = np.std(e_model, axis=0)
    assert np.allclose(e_std, e_std_expect)
    assert np.allclose(e_err, e_std_expect / np.sqrt(10))


def get_locarray(arr:np.ndarray[bhm.h2mm_model], attr:str)->np.ndarray[np.float64]:
    return np.array([getattr(arr[loc], attr)[loc] for loc in product(*(range(n) for n in arr.shape))])


def test_adjust_prior(optsp):
    model = optsp.model
    for r in np.linspace(0,1,12)[1:-1]:
        if r == 0.5: 
            continue
        prior = bhm.error.prior_adjust(model, r, 0)
        if r < 0.5:
            assert prior.prior[0] < model.prior[0]
        else:
            assert prior.prior[0] > model.prior[0]
        prior = bhm.error.prior_adjust(model, r, np.array([False, True, False, False]), np.array([True, True, False, False]))
        assert np.allclose(prior.prior[2:], model.prior[2:])
        if r < 0.5:
            assert prior.prior[1] < model.prior[1]
        else:
            assert prior.prior[1] > model.prior[1]


def test_adjust_trans(optsp):
    model = optsp.model
    loc = np.zeros(model.trans.shape, dtype=np.bool_)
    loc[0,1] = True
    outer = np.zeros(model.trans.shape, dtype=np.bool_)
    outer[0, :2] = True
    for r in np.linspace(0,1,12)[1:-1]:
        if r == 0.5: 
            continue
        trans = bhm.error.trans_adjust(model, r, (0,1))
        if r < 0.5:
            assert trans.trans[0,1] < model.trans[0,1]
        else:
            assert trans.trans[0,1] > model.trans[0,1]
        trans = bhm.error.trans_adjust(model, r, loc, outer)
        assert np.allclose(trans.trans[0,2], model.trans[0,2])
        assert np.allclose(trans.trans[1:,:], model.trans[1:,:])
        if r < 0.5:
            assert trans.trans[0,1] < model.trans[0,1]
        else:
            assert trans.trans[0,1] > model.trans[0,1]


def test_adjust_obs(optsp):
    model = optsp.model
    loc = np.zeros(model.obs.shape, dtype=np.bool_)
    loc[0,1] = True
    outer = np.zeros(model.obs.shape, dtype=np.bool_)
    outer[0, :2] = True
    for r in np.linspace(0,1,12)[1:-1]:
        if r == 0.5: 
            continue
        obs = bhm.error.obs_adjust(model, r, (0,1))
        if r < 0.5:
            assert obs.obs[0,1] < model.obs[0,1]
        else:
            assert obs.obs[0,1] > model.obs[0,1]
        obs = bhm.error.obs_adjust(model, r, loc, outer)
        assert np.allclose(obs.obs[0,2], model.obs[0,2])
        assert np.allclose(obs.obs[1:,:], model.obs[1:,:])
        if r < 0.5:
            assert obs.obs[0,1] < model.obs[0,1]
        else:
            assert obs.obs[0,1] > model.obs[0,1]


def test_evalutate_ll_error_prior(hp3, optsp):
    spdict = optsp.tp.sort_photons(hp3, optsp)
    indexes, times = spdict['indexes'], spdict['times']
    plow, phigh = bhm.error.evalutate_ll_error(optsp.model, indexes, times, bhm.error.prior_adjust, loc=0)
    assert plow.prior[0] < optsp.model.prior[0]
    assert optsp.model.prior[0] < phigh.prior[0]

    
def test_evalutate_ll_error_trans(hp3, optsp):
    spdict = optsp.tp.sort_photons(hp3, optsp)
    indexes, times = spdict['indexes'], spdict['times']
    tlow, thigh = bhm.error.evalutate_ll_error(optsp.model, indexes, times, bhm.error.trans_adjust, loc=(0,1))
    assert tlow.trans[0,1] < optsp.model.trans[0,1]
    assert optsp.model.trans[0,1] < thigh.trans[0,1]
    assert np.allclose(optsp.model.trans[1:,:], tlow.trans[1:,:])
    assert np.allclose(optsp.model.trans[1:,:], thigh.trans[1:,:])


def test_evalutate_ll_error_obs(hp3, optsp):
    spdict = optsp.tp.sort_photons(hp3, optsp)
    indexes, times = spdict['indexes'], spdict['times']
    olow, ohigh = bhm.error.evalutate_ll_error(optsp.model, indexes, times, bhm.error.obs_adjust, loc=(0,0))
    assert olow.obs[0,0] < optsp.model.obs[0,0]
    assert optsp.model.obs[0,0] < ohigh.obs[0,0]
    assert np.allclose(optsp.model.obs[1:,:], olow.obs[1:,:])
    assert np.allclose(optsp.model.obs[1:,:], ohigh.obs[1:,:])


def test_statepath_ll_prior(hp3, optsp):
    plow, phigh = bhm.error.statepath_ll_error(hp3, optsp, adjust='prior')
    assert np.all(plow < phigh)
    plow, phigh = bhm.error.statepath_ll_error(hp3, optsp, adjust='prior', loc=(0,))
    assert plow.prior[0] < phigh.prior[0]


def test_statepath_ll_trans(hp3, optsp):
    tlow, thigh = bhm.error.statepath_ll_error(hp3, optsp, adjust='trans')
    assert np.all(tlow < thigh)
    tlow, thigh = bhm.error.statepath_ll_error(hp3, optsp, adjust='trans', loc=(0,1))
    assert tlow.trans[0,1] < thigh.trans[0,1]


def test_statepath_ll_obs(hp3, optsp):
    olow, ohigh = bhm.error.statepath_ll_error(hp3, optsp, adjust='obs')
    assert np.all(olow < ohigh)
    olow, ohigh = bhm.error.statepath_ll_error(hp3, optsp, adjust='obs', loc=(0,1))
    assert olow.obs[0,1] < ohigh.obs[0,1]

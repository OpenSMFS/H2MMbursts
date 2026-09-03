#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:08:08 2026

@author: paul
"""
from itertools import product

import numpy as np

import smfbursts as smf
import H2MMbursts as bhm

import pytest


bhm.optimization_limits.squarem = True

@pytest.fixture
def simsp(statepath):
    params = statepath.params.asdict
    params['seed'] = 327
    return smf.Param(bhm.sim.H2MMSim, params, statepath.parents)


@pytest.fixture
def dwsimsp(simsp):
    return smf.Param(bhm.sim.SimDwells, statepath=simsp)


@pytest.fixture
def simntsp(ntstatepath):
    params = ntstatepath.params.asdict
    params['seed'] = 327
    return smf.Param(bhm.sim.ntdivH2MMSim, params, ntstatepath.parents)


@pytest.fixture
def dwsimntsp(simntsp):
    return smf.Param(bhm.sim.SimDwells, statepath=simntsp)


@pytest.fixture
def simussp(usstatepath):
    params = usstatepath.params.asdict
    params['seed'] = 327
    return smf.Param(bhm.sim.usAlexH2MMSim, params, usstatepath.parents)


@pytest.fixture
def dwsimussp(simussp):
    return smf.Param(bhm.sim.SimDwells, statepath=simussp)


@pytest.fixture(params=(('istart', ), ('istop', ), ('start', ), ('stop', ), 
                        ('istarttime', ), ('istoptime', ), ('dur', ),
                        ) +
                        tuple(product(('midtime', 'sep'), ('start', 'istarttime'), ('stop', 'istoptime')))
                )
def startstopcol(request):
    return request.param

@pytest.fixture(params=(('nph_raw', smf.PhSel('0ex0em')), 
                        ('ratio_raw', smf.PhSel('0ex1em'), smf.PhSel('0ex')), 
                        ('anisotropy_raw', smf.PhSel('0ex1em'), smf.PhSel('0ex0em')), ) +
                        tuple(product(('brightness',), (smf.PhSel('0ex0em'), ), 
                                      ('start', 'istarttime'), ('stop', 'istoptime')))
                )
def simcol(request):
    return request.param


def test_sim_starstop(hp3, simsp, startstopcol):
    sim = hp3.get_table(simsp)
    base = hp3.get_table(simsp.parents['bursts'])
    assert np.all(sim[startstopcol] == base[startstopcol])


def test_simnt_starstop(hp3, simntsp, startstopcol):
    sim = hp3.get_table(simntsp)
    base = hp3.get_table(simntsp.parents['bursts'])
    assert np.all(sim[startstopcol] == base[startstopcol])


def test_simus_starstop(yopo, simussp, startstopcol):
    sim = yopo.get_table(simussp)[0]
    base = yopo.get_table(simussp.parents['bursts'])[0]
    assert np.all(sim[startstopcol] == base[startstopcol])


def test_sim_newcol(hp3, simsp, simcol):
    sim = hp3.get_table(simsp)
    base = hp3.get_table(simsp.parents['bursts'])
    assert (sim[simcol] == base[simcol]).sum() < 100


def test_simnt_newcol(hp3, simntsp, simcol):
    sim = hp3.get_table(simntsp)
    base = hp3.get_table(simntsp.parents['bursts'])
    assert (sim[simcol] == base[simcol]).sum() < 100


def test_simus_newcol(yopo, simussp, simcol):
    sim = yopo.get_table(simussp)[0]
    base = yopo.get_table(simussp.parents['bursts'])[0]
    assert (sim[simcol] == base[simcol]).sum() < 100


def test_simdwell(hp3, dwsimsp, dwellsp):
    sdw = hp3.get_table(dwsimsp)
    dw = hp3.get_table(dwellsp)
    assert sdw['nph_raw', smf.PhSel('0ex0em')].size != dw['nph_raw', smf.PhSel('0ex0em')].size


def test_simdwellnt(hp3, dwsimntsp, dwellntsp):
    sdw = hp3.get_table(dwsimntsp)
    dw = hp3.get_table(dwellntsp)
    assert sdw['nph_raw', smf.PhSel('0ex0em')].size != dw['nph_raw', smf.PhSel('0ex0em')].size


def test_simdwellus(yopo, dwsimussp, dwellussp):
    sdw = yopo.get_table(dwsimussp)[0]
    dw = yopo.get_table(dwellussp)[0]
    assert sdw['nph_raw', smf.PhSel('0ex0em')].size != dw['nph_raw', smf.PhSel('0ex0em')].size

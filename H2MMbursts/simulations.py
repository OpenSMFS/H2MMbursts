#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Paul David Harris
# Purpose: Simulated data of h2mm models.
"""
Simulations
===========

Module for creating Monte-Carlo simulated data (essentially photon re-assignment)
based on |H2MM| models.

.. |H2MM| replace:: H\ :sup:`2`\ MM
"""
from typing import Any
from collections.abc import Iterator, Iterable, Sequence
from itertools import combinations
from functools import partial
from numbers import Real

import numpy as np

import H2MM_C as hm

from smfbursts.datamodel.utils import _echo, tupledict
from smfbursts.datamodel.immutabledata import TV_int
from smfbursts.datamodel.tables import ParentDef, ParamDef, ColumnDef, Param, Column, paramproperty
from smfbursts.cite import cite
from smfbursts.ph_sel import PhSel, DetDef, phsel_union, PhStream, phsel_all, phsel_none
from smfbursts.photondata import (
    BasePhotonTableLike, BasePhotonTable, ChildPhotonTable, make_base_column_defs, 
    PhotonData, PhotonDataS, _title_sels, _title_unit_append, _pol_ps
    )
import smfbursts.cfuncs as smc

from .modeltables import (
    StatePathBase, StatePath, ntdivStatePath, usAlexStatePath,
    _mask_expand, _astype, make_h2mm_columndefs
    )

from .modeltables import Dwells

def _all_stream_comb(streams:Iterable[PhSel], detdef:DetDef):
    """Iterator over all possible combinations of streams in streams"""
    for r in range(1, len(streams)+1):
        for cmb in  combinations(streams, r):
            stream = cmb[0]
            for c in cmb:
                stream |= c
            yield stream.render_positive(detdef, convert_all=True)


def _check_phsel_instream(phsel:Any, detdef:DetDef, streams:Sequence[PhSel])->Any:
    """
    If not a PhSel of PhStream, return unchanged, otherwise 
    check if phsel is valid for detdef and streams
    """
    phsel = PhSel(phsel) if isinstance(phsel, PhStream) else phsel
    if not isinstance(phsel, PhSel):
        return phsel
    if any((phsel & sel) not in (phsel_none, sel) for sel in streams): 
        raise ValueError("invalid phsel for defined streams of H2MM simulation")
    return phsel


def _regularize_phsel(keytup:tuple, detdef:DetDef, streams:Sequence[PhSel])->tuple:
    """Make all PhSel positive and check if in streams, non-phsel unchanged"""
    return tuple(_check_phsel_instream(key, detdef, streams) for key in keytup)


h2mm_seed_paramdef = ParamDef('seed', TV_int(mn=0, mx=1<<31))


class H2MMSimBase(StatePathBase, ChildPhotonTable):
    """
    Base class for H2MMSim simulation tables.
    This class generates a Monte-Carlo simulted statepath with the transition probabilities
    of the ``model`` parameter. The times are defined by the source data 
    (i.e. photon times are unchanged from source data), while detectors are
    Monte-Carlo assigned based on the model.
    
    Subclasses each replicate the photon processing of a sublcass of 
    :class:`H2MMbursts.modeltables.StatePathBase` the parameters will be those
    of the given StatePath, with the addition of a 32 bit seed value for the
    random number generator.
    
    H2MMSimBase defines ``parent_defs`` and ``column_defs`` but the ``param_defs``
    class attribute must be specified.
    
    ``H2MMSimBase`` is a ``smfbursts.photondata.BasePhotonTableLike`` that is,
    while it is not a base-table, it can be treated as one in certain cases
    as the parent for other tables. In particular this allows ``smfbursts.bursttable.NphBG``
    to treat a H2MMSimBase as a parent, where the photon counts are over-ridden
    with the simulated photon counts (as though the photons were re-indexed).
    
    This means that ``H2MSimBase`` subclasses define the essential columns for
    a base photon table. All non-nanotime related columns are defined. However,
    accessing photon strems that are not composable as unions of the ``streams``
    parameter are disallowed.
    
    .. note::
        
        Use :class:`SimDwell` to examine dwells, as the :class:`H2MMbursts.modeltables.Dwells`
        will use the original, and not simulated detectors from data instead of
        appropriately using the simulated statepath detectors etc.
    
    
    Parents
    -------
    
        bursts : BasePhotonTableLike
            The photon ranges defining the photons to Monte-Carlo reassign detectors.
    
    Columns
    -------
    
        start : int,  ()
            Start time of "bursts" replicates the bursts parent
        istart: int, ()
            Time of first photon in "bursts", replicates bursts parent
        stop : int, ()
            Stop time of "bursts" replicates the bursts parent
        istop : int, ()
            Time of last photon in "bursts", replicates bursts parent
        ph_times : np.ndarray[np.int64], (phsel:PhSel, )
            Times of photons in phsel in bursts
        ph_dets : np.ndarray[np.uint8], (phsel:PhSel, )
            Detector indexes of photon in phsel in bursts. If the streams defines
            streams that cover multiple detectors, detetor id will be the "first"
            detector in the sequence from ``DetDef.get_stream_ids``
        detpath : np.ndarray[np.uint8], ()
            Detector indexes of photons used in |H2MM| processing, not reassigned by streams.
            Each row is uint8 array.
        nph_raw : int, (phsel:PhSel, )
            Number of photons of phsel in each bursts, based on simulation.
        ratio_raw : float (phsel_num:PhSel, phsel_dem:PhSel)
            Ratio of number of photons in phsel_num to phsel_dem based on simulation.
        anisotropy_raw : float (phsel_p:PhSel, phsel_s)
            Anistropy of simulated data between phsel_p (parallel) and phsel_s (perpendicular)
        bva : float (phsel_num:PhSel, phsel_dem:PhSel, n:int)
            burst variance of simulated data.
        ebva : float (phsel_num:PhSel, phsel_dem:PhSel, n:int)
            excess burst variance of simulated data
    
    Remapped Columns
    ----------------
        
        E_raw float, ()
            Raw ratio of nph of PhSel('0ex1em') and PhSel('0ex') of simulated data
        S_raw float, ()
            Raw ratio of nph of PhSel('0ex') and PhSel('0ex_1ex1em') of simulated data
        
    """
    parent_defs = (ParentDef('bursts', BasePhotonTableLike, is_base=True),)
    column_defs = (
        ColumnDef('start', tuple(), 0, 'never', dtype=np.int64, title='start', unit='clk_p', get_func='_get_start'),
        ColumnDef('istart', tuple(), 0, 'never', dtype=np.int64, title='istart', get_func='_get_istart'),
        ColumnDef('stop', tuple(), 0, 'never', dtype=np.int64, title='stop', unit='clk_p', get_func='_get_stop'),
        ColumnDef('istop', tuple(), 0, 'never', dtype=np.int64, title='istop', get_func='_get_istop'),
        ColumnDef('ph_times', (PhSel, ), 0, 'never', get_func='_get_ph_times', 
                  get_derived=False, dtype=np.object_, typedef=np.dtype('<i1')), 
        ColumnDef('ph_dets', (PhSel, ), 0, 'never', iter_func='_iter_ph_dets', 
                  get_derived=False, dtype=np.object_, typedef=np.dtype('<u1')),
        ColumnDef('ph_mask', (PhSel, ),0, 'never', iter_func='_iter_ph_mask',
                  get_derived=False, dtype=np.object_, typedef=np.bool_),
        ColumnDef('detpath', tuple(), 0, 'never', iter_func='_iter_detpath',
                  get_derived=False, dtype=np.object_, typedef=np.dtype('<u1')),
        ColumnDef('bva', (PhSel, PhSel, int), 0, 'user', get_func='_get_bva', 
                  dtype=np.dtype('<f8'), get_derived=True, reg_func='_regularizecolumn_bva',
                  title_func='_get_bva_title'),
            ) + (
                make_h2mm_columndefs(skip=('detpath', 'eff_state')) + 
                make_base_column_defs(skip=('start','istart','stop','istop', 
                                            'ph_time', 'ph_dets', 'ph_nanos', 
                                            'ph_particles', 'ph_mask', 
                                            'bva', 'max_rate', 'nanohist', 'nanomean'))
                 )

    def __init_columns__(self):
        model, seed = self.param.params['model'], self.param.params['seed']
        sort_photons = self._sort_photons(self.origin, bursts=self.param.parents['bursts'], 
                                          **{k:v for k, v in self.param.params.items() 
                                             if k != 'model'})
        times = sort_photons['times']
        states = np.empty(times.shape, dtype=np.object_)
        indexes = np.empty(times.shape, dtype=np.object_)
        for i in range(times.size):
            stemp, itemp = hm.sim_phtraj_from_times(model, times[i], seed=seed)
            seed = None # seed set on fist iteration, after is None so remebmer last seed
            states[i] = stemp
            indexes[i] = itemp
        self._add_column('statepath', tuple(), states)
        self._add_column('indexpath', tuple(), indexes)

    @classmethod
    def _sort_photons(cls, origin:PhotonDataS, statepath:Param=None, bursts:Param=None, 
                      **kwargs)->dict[str:np.ndarray[np.ndarray]]:
        if statepath is not None:
            pdict = statepath.params.asdict
            pdict.pop('seed')
            statepath = Param(cls._sim_class, params=pdict, parents=statepath.parents)
        return cls._sim_class._sort_photons(origin, statepath=statepath, bursts=bursts, **kwargs)

    @classmethod
    def _regularize_column_kwargs(cls, **kwargs)->dict[str,Any]:
        detdef = kwargs['source_param'].detdef
        streams = kwargs['source_param'].params['streams']
        kwargs['keytup'] = _regularize_phsel(kwargs['keytup'], detdef, streams)
        return kwargs

    @classmethod
    def _valid_stream(cls, param:Param, stream:PhSel)->bool:
        streams = param.params['streams']
        detdef = param.detdef
        stream = stream.render_positive(detdef, convert_all=True)
        for cmb in _all_stream_comb(streams, detdef):
            if stream == cmb:
                return True
        return False

    @paramproperty
    def detdef(cls,  param:Param)->DetDef:
        return param.parents['bursts'].detdef

    def phsel_select(self, phsel:PhSel, col:str, fill:Any, dtype:np.dtype)->np.ndarray[np.object_]:
        out = np.empty(self.size, dtype=np.object_)
        arr_type = _echo if dtype is None else partial(_astype, dtype)
        det_map = self.param_idx_to_det_map(self.param.params.asdict, self.origin.detdef)
        dets = self.origin.detdef.get_stream_ids(phsel)
        valid_ids = np.argwhere(np.isin(det_map, dets))
        for i, (arr, idx) in enumerate(zip(self.iter_column(col), self.iter_column('indexpath'))):
            out[i] = arr_type(arr[np.isin(idx, valid_ids)])
        return out

    def _iter_detpath(self)->np.ndarray[np.ndarray[np.uint8]]:
        detdef = self.param.detdef
        det_map = np.array([detdef.get_stream_ids(phsel)[0] for phsel in 
                            self.param.params['streams']], dtype=np.dtype('<u1'))
        for i, idx in enumerate(self['indexpath']):
            yield det_map[idx]

    def _get_start(self)->np.ndarray[np.int64]:
        return self.parents['bursts']['start']

    def _get_istart(self)->np.ndarray[np.int64]:
        return self.parents['bursts']['istart']

    def _get_stop(self)->np.ndarray[np.int64]:
        return self.parents['bursts']['stop']

    def _get_istop(self)->np.ndarray[np.int64]:
        return self.parents['bursts']['istop']

    def _iter_ph_mask(self, phsel:PhSel)->np.ndarray[np.bool_]:        
        didx = self.origin.detdef.get_stream_ids(phsel)
        for ms, idx in zip(self.parents['bursts'].iter_column('ph_mask', self.phsel_span), 
                           self.iter_column('ph_dets', self.phsel_span)):
            mask = np.isin(idx, didx)
            yield _mask_expand(mask, ms, np.ones(ms.size, dtype=np.bool_), False)

    def _get_ph_times(self, phsel:PhSel)->np.ndarray[np.int64]:
        return self.phsel_select(phsel, 'timepath', -1, np.dtype('<i8'))

    def _iter_ph_dets(self, phsel:PhSel)->np.ndarray[np.uint8]:
        det_map = self.param_idx_to_det_map(self.param.params.asdict, self.origin.detdef)
        for idx in  self.phsel_select(phsel, 'indexpath', (1<<8)-1, np.dtype('<u1')):
            yield det_map[idx]

    _get_istarttime = BasePhotonTable._get_istarttime
    _get_istoptime = BasePhotonTable._get_istoptime
    _iter_meanT = BasePhotonTable._iter_meanT
    _iter_mTdiff = BasePhotonTable._iter_mTdiff
    _regularizecolumn_middur = BasePhotonTable._regularizecolumn_middur
    _get_dur = BasePhotonTable._get_dur
    _get_dur_title = BasePhotonTable._get_dur_title
    _get_midtime = BasePhotonTable._get_midtime
    _get_midtime_title = BasePhotonTable._get_midtime_title
    _get_meanT_title = BasePhotonTable._get_meanT_title
    _get_sep = BasePhotonTable._get_sep
    _get_sep_title = BasePhotonTable._get_sep_title
    _regularizecolumn_sep = BasePhotonTable._regularizecolumn_sep
    _get_brightness = BasePhotonTable._get_brightness
    _get_brightness_title = BasePhotonTable._get_brightness_title
    _regularizecolumn_brightness = BasePhotonTable._regularizecolumn_brightness
    _iter_nph_raw = BasePhotonTable._iter_nph_raw
    _get_nph_raw_title = BasePhotonTable._get_nph_raw_title
    _get_ratio_raw = BasePhotonTable._get_ratio_raw
    _get_ratio_raw_title = BasePhotonTable._get_ratio_raw_title
    _replace_E_raw = BasePhotonTable._replace_E_raw
    _replace_S_raw = BasePhotonTable._replace_S_raw
    _get_anisotropy_raw = BasePhotonTable._get_anisotropy_raw
    _get_anisotropy_raw_title = BasePhotonTable._get_anisotropy_raw_title

    @cite('TorellaBioPhyJ2011', purpose='Burst Variance Analysis')
    def _get_bva(self, phsel_num:PhSel, phsel_dem:PhSel, n:int)->np.ndarray[np.float64]:
        stream_idsSub = self.origin.detdef.get_stream_ids(phsel_num)
        stream_idsAll = self.origin.detdef.get_stream_ids(phsel_dem)
        detpath = self['detpath']
        startstop = np.cumsum([det.size for det in detpath])
        startstop = np.concatenate([[0], startstop])
        return smc.burst_variance_analysis(np.concatenate(detpath), 
                                           startstop[:-1], startstop[1:], 
                                           stream_idsAll, stream_idsSub, n=n)

    _get_bva_title = BasePhotonTable._get_bva_title
    _regularizecolumn_bva = BasePhotonTable._regularizecolumn_bva
    _get_ebva = BasePhotonTable._get_ebva
    _get_ebva_title = BasePhotonTable._get_ebva_title

    
class H2MMSim(H2MMSimBase):
    """
    :class:`H2MMSimBase` for basic :class:`H2MMbursts.modeltables.StatePath`
    
    .. note::
        
        Use :class:`SimDwell` to examine dwells, as the :class:`H2MMbursts.modeltables.Dwells`
        will use the original, and not simulated detectors from data instead of
        appropriately using the simulated statepath detectors etc.
    
    
    Params
    ------
        
        model : hm.h2mm_model
            The :class:`hm.h2mm_model` used in *Viterbi* processing.
        streams : tuple[PhSel, ...]
            tuple of :class:`PhSel` defining the indexes of photons in |H2MM| processing.
        seed : int
            Positive integer used as seed of random number generator.
    
    
    Parents
    -------
    
        bursts : BasePhotonTableLike
            The photon ranges defining the photons to Monte-Carlo reassign detectors.
    
    
    Columns
    -------
    
        start : int,  ()
            Start time of "bursts" replicates the bursts parent
        istart: int, ()
            Time of first photon in "bursts", replicates bursts parent
        stop : int, ()
            Stop time of "bursts" replicates the bursts parent
        istop : int, ()
            Time of last photon in "bursts", replicates bursts parent
        ph_times : np.ndarray[np.int64], (phsel:PhSel, )
            Times of photons in phsel in bursts
        ph_dets : np.ndarray[np.uint8], (phsel:PhSel, )
            Detector indexes of photon in phsel in bursts. If the streams defines
            streams that cover multiple detectors, detetor id will be the "first"
            detector in the sequence from ``DetDef.get_stream_ids``
        detpath : np.ndarray[np.uint8], ()
            Detector indexes of photons used in |H2MM| processing, not reassigned by streams.
            Each row is uint8 array.
        nph_raw : int, (phsel:PhSel, )
            Number of photons of phsel in each bursts, based on simulation.
        ratio_raw : float (phsel_num:PhSel, phsel_dem:PhSel)
            Ratio of number of photons in phsel_num to phsel_dem based on simulation.
        anisotropy_raw : float (phsel_p:PhSel, phsel_s)
            Anistropy of simulated data between phsel_p (parallel) and phsel_s (perpendicular)
        bva : float (phsel_num:PhSel, phsel_dem:PhSel, n:int)
            burst variance of simulated data.
        ebva : float (phsel_num:PhSel, phsel_dem:PhSel, n:int)
            excess burst variance of simulated data
    
    Remapped Columns
    ----------------
        
        E_raw float, ()
            Raw ratio of nph of PhSel('0ex1em') and PhSel('0ex') of simulated data
        S_raw float, ()
            Raw ratio of nph of PhSel('0ex') and PhSel('0ex_1ex1em') of simulated data
    
    
    """
    _sim_class = StatePath
    param_defs = StatePath.param_defs + (h2mm_seed_paramdef, ) #: :meta private:

    param_idx_to_det_map = StatePath.param_idx_to_det_map
    get_ndet = StatePath.get_ndet
    model_streams = StatePath.model_streams
    model_value = StatePath.model_value
    validate_param = StatePath.validate_param


class ntdivH2MMSim(H2MMSimBase):
    """
    :class:`H2MMSimBase` for basic :class:`H2MMbursts.modeltables.ntdivStatePath`
    
    Columns based on nanotimes (nanomean) are not available, however, model values
    of such columns are available.
    
    .. note::
        
        Use :class:`SimDwell` to examine dwells, as the :class:`H2MMbursts.modeltables.Dwells`
        will use the original, and not simulated detectors from data instead of
        appropriately using the simulated statepath detectors etc.
    
    Params
    ------
        model : hm.h2mm_model
            The :class:`hm.h2mm_model` used in *Viterbi* processing.
        streams : tuple[PhSel, ...]
            tuple of :class:`PhSel` defining the indexes of photons in |H2MM| processing.
        divs : tuple[np.ndarray[np.int16],...]
            tuple of arrays of positions of divisors per photon stream. Divisors
            set by raw nanotime. For each stream there will be 1 more indexes
            compared to the size of the cooresponding divs array
        seed : int
            Positive integer used as seed of random number generator.
    
    Columns
    -------
    
        start : int,  ()
            Start time of "bursts" replicates the bursts parent
        istart: int, ()
            Time of first photon in "bursts", replicates bursts parent
        stop : int, ()
            Stop time of "bursts" replicates the bursts parent
        istop : int, ()
            Time of last photon in "bursts", replicates bursts parent
        ph_times : np.ndarray[np.int64], (phsel:PhSel, )
            Times of photons in phsel in bursts
        ph_dets : np.ndarray[np.uint8], (phsel:PhSel, )
            Detector indexes of photon in phsel in bursts. If the streams defines
            streams that cover multiple detectors, detetor id will be the "first"
            detector in the sequence from ``DetDef.get_stream_ids``
        detpath : np.ndarray[np.uint8], ()
            Detector indexes of photons used in |H2MM| processing, not reassigned by streams.
            Each row is uint8 array.
        nph_raw : int, (phsel:PhSel, )
            Number of photons of phsel in each bursts, based on simulation.
        ratio_raw : float (phsel_num:PhSel, phsel_dem:PhSel)
            Ratio of number of photons in phsel_num to phsel_dem based on simulation.
        anisotropy_raw : float (phsel_p:PhSel, phsel_s)
            Anistropy of simulated data between phsel_p (parallel) and phsel_s (perpendicular)
        bva : float (phsel_num:PhSel, phsel_dem:PhSel, n:int)
            burst variance of simulated data.
        ebva : float (phsel_num:PhSel, phsel_dem:PhSel, n:int)
            excess burst variance of simulated data
    
    Remapped Columns
    ----------------
        
        E_raw float, ()
            Raw ratio of nph of PhSel('0ex1em') and PhSel('0ex') of simulated data
        S_raw float, ()
            Raw ratio of nph of PhSel('0ex') and PhSel('0ex_1ex1em') of simulated data
    
    """
    _sim_class = ntdivStatePath
    param_defs = ntdivStatePath.param_defs + (h2mm_seed_paramdef, ) #: :meta private:

    param_idx_to_det_map = ntdivStatePath.param_idx_to_det_map
    get_ndet = ntdivStatePath.get_ndet
    model_streams = ntdivStatePath.model_streams
    model_value = ntdivStatePath.model_value
    validate_param = ntdivStatePath.validate_param


class usAlexH2MMSim(H2MMSimBase):
    """
    :class:`H2MMSimBase` for basic :class:`H2MMbursts.modeltables.usAlexStatePath`
    
    .. note::
        
        Use :class:`SimDwell` to examine dwells, as the :class:`H2MMbursts.modeltables.Dwells`
        will use the original, and not simulated detectors from data instead of
        appropriately using the simulated statepath detectors etc.
    
    
    Params
    ------
        model : hm.h2mm_model
            The :class:`hm.h2mm_model` used in *Viterbi* processing.
        streams : tuple[PhSel, ...]
            tuple of :class:`PhSel` defining the indexes of photons in |H2MM| processing.
        shifts : tuple[str, ...]
            String describing how to shift (which shift function to use) the given
            stream, tuple must be same length as streams.
        seed : int
            Positive integer used as seed of random number generator.
    
    Columns
    -------
    
        start : int,  ()
            Start time of "bursts" replicates the bursts parent
        istart: int, ()
            Time of first photon in "bursts", replicates bursts parent
        stop : int, ()
            Stop time of "bursts" replicates the bursts parent
        istop : int, ()
            Time of last photon in "bursts", replicates bursts parent
        ph_times : np.ndarray[np.int64], (phsel:PhSel, )
            Times of photons in phsel in bursts
        ph_dets : np.ndarray[np.uint8], (phsel:PhSel, )
            Detector indexes of photon in phsel in bursts. If the streams defines
            streams that cover multiple detectors, detetor id will be the "first"
            detector in the sequence from ``DetDef.get_stream_ids``
        detpath : np.ndarray[np.uint8], ()
            Detector indexes of photons used in |H2MM| processing, not reassigned by streams.
            Each row is uint8 array.
        nph_raw : int, (phsel:PhSel, )
            Number of photons of phsel in each bursts, based on simulation.
        ratio_raw : float (phsel_num:PhSel, phsel_dem:PhSel)
            Ratio of number of photons in phsel_num to phsel_dem based on simulation.
        anisotropy_raw : float (phsel_p:PhSel, phsel_s)
            Anistropy of simulated data between phsel_p (parallel) and phsel_s (perpendicular)
        bva : float (phsel_num:PhSel, phsel_dem:PhSel, n:int)
            burst variance of simulated data.
        ebva : float (phsel_num:PhSel, phsel_dem:PhSel, n:int)
            excess burst variance of simulated data
    
    Remapped Columns
    ----------------
        
        E_raw float, ()
            Raw ratio of nph of PhSel('0ex1em') and PhSel('0ex') of simulated data
        S_raw float, ()
            Raw ratio of nph of PhSel('0ex') and PhSel('0ex_1ex1em') of simulated data
    
    """
    _sim_class = usAlexStatePath
    param_defs = usAlexStatePath.param_defs + (h2mm_seed_paramdef, ) #: :meta private:
    #: :meta private:
    column_defs = H2MMSimBase.column_defs + (
        ColumnDef('sortpath', tuple(), 0, 'all', dtype=np.object_, typedef=np.dtype('<i8')), )
    
    param_idx_to_det_map = usAlexStatePath.param_idx_to_det_map #: :meta private:
    get_ndet = usAlexStatePath.get_ndet
    model_streams = usAlexStatePath.model_streams
    model_value = usAlexStatePath.model_value
    validate_param = usAlexStatePath.validate_param
    
    def __init_columns__(self):
        model, seed = self.param.params['model'], self.param.params['seed']
        sort_photons = self._sort_photons(self.origin, bursts=self.param.parents['bursts'], 
                                          **{k:v for k, v in self.param.params.items() 
                                             if k != 'model'})
        times = sort_photons['times']
        states = np.empty(times.shape, dtype=np.object_)
        indexes = np.empty(times.shape, dtype=np.object_)
        for i in range(times.size):
            stemp, itemp = hm.sim_phtraj_from_times(model, times[i], seed=seed)
            seed = None # seed set on fist iteration, after is None so remebmer last seed
            states[i] = stemp
            indexes[i] = itemp
        self._add_column('statepath', tuple(), states)
        self._add_column('indexpath', tuple(), indexes)
        self._add_column('sortpath', tuple(), indexes)
    
    @paramproperty
    def mirror_param(cls, param:Param):
        pdict = param.params.asdict
        pdict.pop('seed')
        return Param(cls._sim_class, pdict, param.parents)
    
    def phsel_select(self, phsel, col, fill, dtype):
        out = np.empty(self.size, dtype=np.object_)
        arr_type = _echo if dtype is None else partial(_astype, dtype)
        det_map = self.param_idx_to_det_map(self.param.params.asdict, self.origin.detdef)
        dets = self.origin.detdef.get_stream_ids(phsel)
        valid_ids = np.argwhere(np.isin(det_map, dets))
        for i, (c, idx, sort) in enumerate(zip(self.iter_column(col), 
                                                 self.iter_column('indexpath'), 
                                                 self.iter_column('sortpath'))):
            out[i] = arr_type(c[np.isin(idx, valid_ids)])
        return out


class SimDwells(Dwells):
    """
    Dwells of a Monte-Carlo Simulation 
    (:class:`H2MMSim`, :class:`ntdivH2MMSim` or :class:`usAlexH2MMSim`)
    
    
    Parents
    -------
        
        statepath : H2MMSimBase
            Simulated statepath
    """
    parent_defs = (ParentDef('statepath', H2MMSimBase), ) #: :meta private:
    #: :meta private:
    column_defs = tuple(cdef for cdef in Dwells.column_defs if 'nano' not in cdef.name)
    _parent_ph_subrange = 'statepath'
    
    def _iter_ph_array(self, key:str, phsel:PhSel):
        sim = self.parents['statepath']
        biter = zip(sim.iter_column('ph_times', phsel), sim.iter_column(key, phsel))
        diter = zip(self.iter_column('iburst'), self.iter_column('start'), self.iter_column('stop'))
        bi, dstart, dstop = next(diter)
        for curb, (times, arr) in enumerate(biter):
            while curb == bi:
                mask = (dstart <= times) & (times < dstop)
                yield arr[mask]
                try:
                    bi, dstart, dstop = next(diter)
                except StopIteration:
                    break

    def _iter_ph_mask(self, phsel:PhSel)->np.ndarray[np.bool_]:
        sim = self.parents['statepath']
        biter = zip(sim.iter_column('ph_times', phsel), sim.iter_column('ph_dets', phsel))
        diter = zip(self.iter_column('iburst'), self.iter_column('start'), self.iter_column('stop'))
        bi, dstart, dstop = next(diter)
        dids = self.origin.detdef.get_stream_ids(phsel)
        for curb, (times, dets) in enumerate(biter):
            dmask = np.isin(dets, dids)
            while curb == bi:
                mask = (dstart <= times) & (times < dstop)
                yield dmask[mask]
                try:
                    bi, dstart, dstop = next(diter)
                except StopIteration:
                    break

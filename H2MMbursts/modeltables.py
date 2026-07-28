#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Viterbi-based Tables
====================

The ``modeltables`` module defines FRETBursts tables subclasses for |H2MM| processing.

The :class:`StatePathBase` super-set of tables of which the "primary" 
Table is :class:`StatePath`.
This is a child table, the base table is usually a :class:`smfbursts.bursttables.Bursts` 
or :class:`smfbursts.burstables.BurstOvlp` table, and represents
the *Viterbi* most likely path of |H2MM| states through each burst.
These classes also provide :meth:`StatePath.optimize` and :meth:`optimize_models` 
classmethods which allow for optimization of either a single initial model, 
or through a range of numbers of states to generate optimal models and statepaths 
based on a given :class:`smfbursts.photondata.PhotonData`  set of raw data and 
a given burst search. 
:class:`ntdivStatePath` allows incorporation of lifetime data through the 
|divisorapproach| aproach while
:class:`usAlexStatePath` implements the shift method for incorporating the
acceptor excitation stream of :math:`\mu s`\ ALEX experiments into |H2MM| analysis.

The :class:`Dwells` table is a base-table that takes a :class:`StatePathBase` as
it's only parent, and subdivides each burst into dwells in each state.
This makes dwells the rows of the table.


.. |H2MM| replace:: H\ :sup:`2`\ MM
.. |Param| replace:: :class:`Param <smfbursts.datamodel.tables.Param>`
.. |divisorapproach| replace:: `divisor <https://doi.org/10.1016/j.bpr.2022.100071>`__
.. |classmethod| replace:: `classmethod() <https://docs.python.org/3/library/functions.html#classmethod>`__
.. |Kache2005| replace:: `Kache & Hendrix, 2025 <https://doi.org/10.1186/s44330-025-00039-2>`__
"""
from typing import Any, Literal
from collections.abc import Sequence, Iterator, Callable
from itertools import chain, permutations, repeat
from functools import partial
import warnings
from numbers import Integral, Real
import re

import numpy as np

from smfbursts.datamodel.utils import tupledict, fnumba, _echo, ImDict
from smfbursts.datamodel.immutabledata import (
    TypeValidator, TV_tuple, TV_str, TV_ndarray, TV_int, TV_frozenset,
    register_byteslike
    )
from smfbursts.datamodel.tables import (
    ParamDef, ParentDef, ColumnDef, Param, Column, GateGroup, Table, 
    as_paramdict, parammethod, paramproperty, tableproperty
    )
from smfbursts.cite import cite, add_citation
from smfbursts.ph_sel import (
    PhStream, PhSel, DetDef, TV_PhSel, phsel_union, sort_phsels, 
    reindex_phsel, phsel_all
    )
from smfbursts.photondata import (
    PhSpec, PhotonData, PhotonDataList, PhotonDataS, 
    BasePhotonTable, ChildPhotonTable, make_base_column_defs
    )
from smfbursts.childphotontables import NphBG, Ratios

import smfbursts.cfuncs as smc

import H2MM_C as hm

#: Type hint for times array of |H2MM| input
TArray = np.ndarray[np.ndarray[np.int64]]
#: Type hint for dets array of |H2MM| input
DArray = np.ndarray[np.ndarray[np.uint8]]


def _update_dict_new(dct:dict, update:dict):
    """return copy of dct with update"""
    dct = dct.copy()
    dct.update(update)
    return dct


@fnumba.jit(fnumba.bool_(fnumba.uint8[:], fnumba.uint8[:]))
def _anyin(a, b):
    """Determine if any element of a is in b"""
    for i in a:
        if i in b:
            return True
    return False


@fnumba.jit(fnumba.bool_(fnumba.uint8[:], fnumba.uint8[:]))
def _allin(a, b):
    """Determine if all elements of a are in b"""
    for i in a:
        if i not in b:
            return False
    return True


def _check_h2mm_model(val:hm.h2mm_model, nstate:int=None, ndet:int=None, **kwargs)->hm.h2mm_model:
    """
    Check function for TypeValidator of :class:`hm.h2mm_model`

    Parameters
    ----------
    val : hm.h2mm_model
        input model.
    nstate : int, optional
        Necessary number of states in model. The default is None.
    ndet : int, optional
        necessary number of detectors (indexes) in model. The default is None.
    **kwargs : Any
        Ignored.

    Raises
    ------
    TypeError
        Not an h2mm model
    ValueError
        Mismatched nstate or ndet.

    Returns
    -------
    hm.h2mm_model
        immutable version of input h2mm_model.

    """
    if not isinstance(val, hm.h2mm_model):
        raise TypeError("must be h2mm_model, not {type(val)}")
    if nstate is not None and nstate != val.nstate:
        raise ValueError(f'incorrect number of states, expected {nstate} but model has {val.nsdet}')
    if ndet is not None and nstate != val.ndet:
        raise ValueError(f'incorrect number of states, expected {ndet} but model has {val.nsdet}')
    return hm.h2mm_model(val.prior, val.trans, val.obs).sort_states()


def dread_h2mm_model(arr:bytes, dct:dict)->hm.h2mm_model:
    """Read H2MM model from bytes, for TypeValidator"""
    return hm.h2mm_model.frombytes(arr)


def dwrite_h2mm_model(val:hm.h2mm_model)->bytes:
    """Get bytes representation of :class:`hm.h2mm_model` for writing to HDF5 file"""
    return val.tobytes()


def encode_h2mmmodel(val:hm.h2mm_model)->tuple[bytes,]:
    """
    Encode a |H2MM| model object as a tuple for packing into msgpack bytes object.

    Parameters
    ----------
    val : hm.h2mm_model
        |H2MM| model to be stored as tuple in msgpack bytes.

    Returns
    -------
    tuple[bytes,]
        single tuple containing the bytes representation of the |H2MM| model.

    """
    return (val.tobytes(), )


def decode_h2mmmodel(val:tuple[str,bytes])->hm.h2mm_model:
    """
    Decode a tuple from msgpack bytes as an |H2MM| model.

    Parameters
    ----------
    val : tuple[str,bytes]
        msgpack tuple representing |H2MM| model.

    Returns
    -------
    hm.h2mm_model
        |H2MM| model of val.

    """
    return hm.h2mm_model.frombytes(val[1])

#: TypeValidator for |H2MM| models.
#: Uses :func:`check_h2mm_model` as check func.
#: The options are:
#: 
#: - nstate (int) Number of states that must be in the model (the nstate attribute)
#: - ndet (int) Number of detector indexes that must be in the model (the ndet attribute)
#: 
TV_H2MMModel = register_byteslike(hm.h2mm_model, _check_h2mm_model, 
                                  encode_h2mmmodel, decode_h2mmmodel, 
                                  dread_h2mm_model, dwrite_h2mm_model)


def sort_indexes_times(table:BasePhotonTable, streams:Sequence[PhSel])->tuple[np.ndarray[np.uint8],np.ndarray[np.int64]]:
    """
    Get numpy arrays of indices and times for |H2MM| analysis from a base table, 
    and stream definition of streams (tuple of :class:`smfbursts.ph_sel.PhSel`)

    Parameters
    ----------
    table : BasePhotonTable
        Base table defining the bursts over which to create index/time arrays.
    streams : Sequence[PhSel]
        Streams in order of output index over which to produce index/time arrays.

    Returns
    -------
    out_dets : np.ndarray[np.ndarray[np.uint8]]
        Array of array of index for each photon in each burst.
    out_times : np.ndarray[np.ndarray[np.int64]]
        Array of arrays of times for each photon in each burst.

    """
    phselu = phsel_union(*streams)
    id_map = reindex_phsel(table.origin.detdef, streams)
    out_dets = np.empty(table.size, dtype=np.object_)
    out_times = np.empty(table.size, dtype=np.object_)
    if not table.origin.save_memory and hasattr(table.origin, 'times'):
        times, dets = table.origin.times, id_map[table.origin.dets]
        mask = dets != -1
        for i, (istart, istop) in enumerate(zip(table.iter_column('istart'), 
                                                table.iter_column('istop'))):
            out_dets[i] = dets[istart:istop][mask[istart:istop]].astype(np.uint8)
            out_times[i] = times[istart:istop][mask[istart:istop]]
    else:
        for i, (dets, times) in enumerate(zip(table.iter_column('ph_dets', phselu), 
                                    table.iter_column('ph_times', phselu))):
            mask = dets >= 0
            out_dets[i] = id_map[dets[mask]].astype(np.uint8)
            out_times[i] = times[mask]
    return out_dets, out_times


def _mask_expand(arr:np.ndarray, mask_s:np.ndarray[np.bool_], mask_d:np.ndarray[np.bool_], fill:Any)->np.ndarray:
    """Take an array and expand it by the size of mask, filing all False locations of mask with fill"""
    if arr.dtype != np.object_:
        out = np.ones(mask_s.shape+arr.shape[1:], dtype=arr.dtype)*fill
    else:
        out = np.empty(mask_s.shape+arr.shape[1:], dtype=arr.dtype)
        for i in range(mask_s.size):
            out[i] = fill
    out[mask_s] = arr
    return out[mask_d]


def _mask_expand_like(arr:np.ndarray[np.object_], like:np.ndarray[np.object_], 
                      mask:np.ndarray[np.bool_], fill:Any, dtype:np.dtype)->np.ndarray[np.object_]:
    """Mask an expand array by mask (output is size of mask, arr should have
    size of true values of mask, fill values have size of arrays in like"""
    out = np.empty(like.shape, dtype=np.object_)
    out[mask] = arr
    for loc in np.argwhere(~mask):
        out[tuple(loc)] = np.ones(like[tuple(loc)].shape, dtype=dtype)*fill
    return out


def _mask_expand_gamma(arr:np.ndarray[np.object_], like:np.ndarray[np.object_], 
                       mask:np.ndarray[np.bool_], nstate:Any)->np.ndarray[np.object_]:
    """Mask an array by exand into mask, missing arrays have shape of like x nstate"""
    out = np.empty(like.shape, dtype=np.object_)
    out[mask] = arr
    for loc in np.argwhere(~mask):
        out[loc] = np.empty((like.shape[0], nstate), dtype=np.dtype('<f8'))*np.nan
    return out


def _empty_remap(arr:np.ndarray[np.ndarray])->None|np.ndarray[np.bool_]:
    """Create mask of array of arrays, indicating which can be included in H2MM (size >= 2)"""
    mask = np.array([a.size < 2 for a in arr.ravel()]).reshape(arr.shape)
    return ~mask if np.any(mask) else None


def _empty_remove(arr:np.ndarray[np.ndarray])->np.ndarray[np.ndarray]:
    """Remove arrays of size < 2 from arr"""
    mask = _empty_remap(arr)
    return arr if mask is None else arr[mask]


def _viterbi_path(model:hm.h2mm_model, indexes:DArray, times:TArray,
                  **kwargs)->tuple[np.ndarray[np.ndarray[np.uint8]],np.ndarray[np.ndarray[np.float64]],np.ndarray[np.float64],float]:
    """Compute *viterbi* and expand by unused photons, also return scale, loglik and bic"""
    mask = _empty_remap(indexes) # catch any bursts that are too small (< 2 photons) for processing
    if mask is None:
        return hm.viterbi_path(model, indexes, times, **kwargs)
    path, scale, ll, icl = hm.viterbi_path(model, indexes[mask], times[mask], **kwargs)
    # fill with empty arrays
    path = _mask_expand_like(path, indexes, mask, 255, np.dtype('<u1'))
    scale = _mask_expand_like(scale, indexes, mask, np.nan, np.dtype('<f8'))
    ll = _mask_expand(ll, mask, slice(None), 0.0)
    return path, scale, ll, icl


def _h2mm_evaluate(model:hm.h2mm_model, indexes:DArray, times:TArray, **kwargs)->hm.h2mm_model:
    """Evalueate a model, inplace = False"""
    return model.evaluate(_empty_remove(indexes), _empty_remove(times), inplace=False, **kwargs)


def _h2mm_evaluate_gamma(model:hm.h2mm_model, indexes:DArray, times:TArray, **kwargs):
    """Generate gamma array expanded by unused photon stream"""
    mask = _empty_remap(indexes)
    if mask is None:
        _, gamma = hm.H2MM_arr(model, indexes, times, gamma=True, **kwargs)
        return gamma
    _, gamma = hm.H2MM_arr(model, indexes[mask], times[mask], gamma=True, **kwargs)
    return _mask_expand_gamma(gamma, indexes, mask, model.nstate)


def _h2mm_path_loglik(model:hm.h2mm_model, indexes:DArray, times:TArray, state_path:DArray, **kwargs):
    """Generate path-loglik array from model, expanding the ll of each photon 
    with by unused photons in data set, values filled with nan"""
    mask = _empty_remap(indexes)
    if mask is None:
        return hm.path_loglik(model, indexes, times, state_path, BIC=False, loglikpath=True, **kwargs)
    ll = hm.path_loglik(model, indexes[mask], times[mask], state_path[mask], BIC=False, loglikpath=True, **kwargs)
    return _mask_expand_like(ll, indexes, mask, np.nan, np.dtype('<f8'))


if fnumba.has_numba:
    @fnumba.jit(fnumba.int8[:](fnumba.int8[:],fnumba.int64[:]))
    def _infer_state(states:np.ndarray[np.int8], times:np.ndarray[np.int64])->np.ndarray[np.int8]:
        """
        For a given state path where some photons are misssing, infer state of
        missing photons

        Parameters
        ----------
        states : np.ndarray[np.int8]
            State path with -1 values for all photons whose state is to be infered.
        times : np.ndarray[np.int64]
            Arrival times of photons.

        Returns
        -------
        out : np.ndarray[np.int8]
            Infered state path.

        """
        i, bpos, fpos = 0, -1, 0
        out = np.empty(states.size, dtype=np.int8)
        while i < states.size:
            if states[i] != -1:
                out[i] = states[i]
                bpos = i
                i += 1
            else:
                if fpos < i:
                    fpos = i
                while fpos < states.size and states[fpos] == -1:
                    fpos += 1
                if fpos == states.size:
                    for j in range(i, fpos):
                        out[j] = states[bpos]
                else:
                    fwd = False
                    for j in range(i, fpos):
                        if fwd or bpos == -1:
                            out[j] = states[fpos]
                        elif (times[j] - times[bpos]) < (times[fpos] - times[j]):
                            out[j] = states[bpos]
                        else:
                            fwd = True
                            out[j] = states[fpos]
                i = fpos
        return out
    
    
    def infer_state(states:np.ndarray[np.int8], times:np.ndarray[np.int64])->np.ndarray[np.int8]:
        """
        For a given state path where some photons are misssing, infer state of
        missing photons

        Parameters
        ----------
        states : np.ndarray[np.int8]
            State path with -1 values for all photons whose state is to be infered.
        times : np.ndarray[np.int64]
            Arrival times of photons.

        Returns
        -------
        out : np.ndarray[np.int8]
            Infered state path.

        """
        return _infer_state(states if states.flags['WRITEABLE'] else states.copy(),
                            times if times.flags['WRITEABLE'] else times.copy())
    
    
    @fnumba.jit('i8[:,:](u2[:], i1[:], f8[:], f8, i8, i8, u2)', boundscheck=True)
    def _get_nanohist_thresh(nanos:np.ndarray[np.uint16], states:np.ndarray[np.uint8],
                             discr:np.ndarray[np.float64], thresh:float, 
                             nstate:int, ln:int, nanosub:int)->np.ndarray[np.int64]:
        """
        Internal numba optimized function.
        Generate the 2D histogram of states x tcspc bin, given the photon nanotimes
        in ``nanos`` and the *Viterbi* state assignment in ``states``, while filtering
        out photons whose certainty of verterbi assigments 
        (should either be posterior probability or path loglikelihood) is less
        than ``thresh`` (if certainy is greater than or equal to the thresh, photon is included)
        Arguments ln and nstate are the expected number of tcspc bins and states
        respetively, used as not all bursts have nanotimes covering all tcspc bins
        or include all states.

        Parameters
        ----------
        nanos : np.ndarray[np.uint16]
            Nanotimes of burst.
        states : np.ndarray[np.uint8]
            *Viterbi* assigned state of burst.
        discr : np.ndarray[np.float64]
            Statistical likelihood of each state assignement of *Viterbi*.
        thresh : float
            Threshold of discr to include photon in histogram.
        nstate : int
            Number of states.
        ln : int
            Number of TCSPC bins.
        nanosub : int
            Amount to subtract from nanos to set threshold
        
        Returns
        -------
        np.ndarray[np.int64]
            2D histogram, organized (state, tcspc bin).

        """
        out = np.zeros((nstate, ln), dtype=np.int64)
        for i in range(nanos.shape[0]):
            if thresh <= discr[i]:
                if states[i] >= nstate:
                    continue
                out[states[i], nanos[i]-nanosub] += 1
        return out
    
    @fnumba.jit('i8[:,:](u2[:], i1[:], f8[:,:], f8, i8, i8, u2)', boundscheck=True)
    def _get_nanohist_thresh_gamma(nanos:np.ndarray[np.uint16], states:np.ndarray[np.uint8],
                                   discr:np.ndarray[np.float64], thresh:float, 
                                   nstate:int, ln:int, nanosub:int)->np.ndarray[np.int64]:
        """
        Internal numba optimized function.
        Generate the 2D histogram of states x tcspc bin, given the photon nanotimes
        in ``nanos`` and the *Viterbi* state assignment in ``states``, while filtering
        out photons whose certainty of verterbi assigments 
        (should either be posterior probability or path loglikelihood) is less
        than ``thresh`` (if certainy is greater than or equal to the thresh, photon is included)
        Arguments ln and nstate are the expected number of tcspc bins and states
        respetively, used as not all bursts have nanotimes covering all tcspc bins
        or include all states.

        Parameters
        ----------
        nanos : np.ndarray[np.uint16]
            Nanotimes of burst.
        states : np.ndarray[np.uint8]
            *Viterbi* assigned state of burst.
        discr : np.ndarray[np.float64]
            Statistical likelihood of each state assignement of *Viterbi*.
        thresh : float
            Threshold of discr to include photon in histogram.
        nstate : int
            Number of states.
        ln : int
            Number of TCSPC bins.
        nanosub : int
            Amount to subtract from nanos to set threshold
        
        Returns
        -------
        np.ndarray[np.int64]
            2D histogram, organized (state, tcspc bin).

        """
        out = np.zeros((nstate, ln), dtype=np.int64)
        for i in range(nanos.shape[0]):
            if states[i] >= nstate:
                continue
            if thresh <= discr[i, states[i]]:
                out[states[i], nanos[i]-nanosub] += 1
        return out

    
    def get_nanohist_thresh(nanos:np.ndarray[np.uint16], states:np.ndarray[np.uint8],
                            discr:np.ndarray[np.float64], thresh:float, 
                            nstate:int, ln:int, nanosub:int)->np.ndarray[np.int64]:
        """
        Generate the 2D histogram of states x tcspc bin, given the photon nanotimes
        in ``nanos`` and the *Viterbi* state assignment in ``states``, while filtering
        out photons whose certainty of verterbi assigments 
        (should either be posterior probability or path loglikelihood) is less
        than ``thresh`` (if certainy is greater than or equal to the thresh, photon is included)
        Arguments ln and nstate are the expected number of tcspc bins and states
        respetively, used as not all bursts have nanotimes covering all tcspc bins
        or include all states.

        Parameters
        ----------
        nanos : np.ndarray[np.uint16]
            Nanotimes of burst.
        states : np.ndarray[np.uint8]
            *Viterbi* assigned state of burst.
        discr : np.ndarray[np.float64]
            Statistical likelihood of each state assignement of *Viterbi*.
        thresh : float
            Threshold of discr to include photon in histogram.
        nstate : int
            Number of states.
        ln : int
            Number of TCSPC bins.
        nanosub : int
            Amount to subtract from nanos to set threshold

        Returns
        -------
        np.ndarray[np.int64]
            2D histogram, organized (state, tcspc bin).

        """
        return _get_nanohist_thresh(nanos if nanos.flags['WRITEABLE'] else nanos.copy(),
                                    states if states.flags['WRITEABLE'] else states.copy(),
                                    discr if discr.flags['WRITEABLE'] else discr.copy,
                                    thresh, nstate, ln, nanosub)
    
    
    def get_nanohist_thresh_gamma(nanos:np.ndarray[np.uint16], states:np.ndarray[np.uint8],
                                  discr:np.ndarray[np.float64], thresh:float, 
                                  nstate:int, ln:int, nanosub:int)->np.ndarray[np.int64]:
        """
        Generate the 2D histogram of states x tcspc bin, given the photon nanotimes
        in ``nanos`` and the *Viterbi* state assignment in ``states``, while filtering
        out photons whose likelihood (gamma) for the *Viterbi* state is less
        than ``thresh`` (if certainy is greater than or equal to the thresh, photon is included)
        Arguments ln and nstate are the expected number of tcspc bins and states
        respetively, used as not all bursts have nanotimes covering all tcspc bins
        or include all states.

        Parameters
        ----------
        nanos : np.ndarray[np.uint16]
            Nanotimes of burst.
        states : np.ndarray[np.uint8]
            *Viterbi* assigned state of burst.
        discr : np.ndarray[np.float64]
            Photons x states 2D array of likelihood for each state and each photon.
        thresh : float
            Threshold of discr to include photon in histogram.
        nstate : int
            Number of states.
        ln : int
            Number of TCSPC bins.
        nanosub : int
            Amount to subtract from nanos to set threshold

        Returns
        -------
        np.ndarray[np.int64]
            2D histogram, organized (state, tcspc bin).

        """
        return _get_nanohist_thresh_gamma(nanos if nanos.flags['WRITEABLE'] else nanos.copy(),
                                          states if states.flags['WRITEABLE'] else states.copy(),
                                          discr if discr.flags['WRITEABLE'] else discr.copy,
                                          thresh, nstate, ln, nanosub)

else:
    def infer_state(states:np.ndarray[np.int8], times:np.ndarray[np.int64])->np.ndarray[np.int8]:
        """
        For a given state path where some photons are misssing, infer state of
        missing photons

        Parameters
        ----------
        states : np.ndarray[np.int8]
            State path with -1 values for all photons whose state is to be infered.
        times : np.ndarray[np.int64]
            Arrival times of photons.

        Returns
        -------
        out : np.ndarray[np.int8]
            Infered state path.

        """
        out = states.copy()
        mask = states != -1
        rmask = ~mask
        times_defined, states_defined = times[mask], states[mask]
        times_undef = times[rmask]
        idx_high = np.cumsum(mask)[rmask]
        idx_low = idx_high - 1
        idx_low[idx_low == -1] = 0
        times_new_low, states_new_low = times_defined[idx_low], states_defined[idx_low]
        idx_high[idx_high == times_defined.size] -= 1
        times_new_high, states_new_high = times_defined[idx_high], states_defined[idx_high]
        sub_mask_update = (times_undef - times_new_low) < (times_new_high - times_undef)
        mask_update = rmask.copy()
        mask_update[rmask] = sub_mask_update
        out[mask_update] = states_new_low[sub_mask_update]
        sub_mask_update ^= True
        mask_update[rmask] = sub_mask_update
        out[mask_update] = states_new_high[sub_mask_update]
        return out


    def get_nanohist_thresh(nanos:np.ndarray[np.int64], states:np.ndarray[np.int8],
                            discr:np.ndarray[np.float64], thresh:float, 
                            nstate:int, ln:int, nanosub:int)->np.ndarray[np.int64]:
        """
        Generate the 2D histogram of states x tcspc bin, given the photon nanotimes
        in ``nanos`` and the *Viterbi* state assignment in ``states``, while filtering
        out photons whose likelihood (gamma) for the *Viterbi* state is less
        than ``thresh`` (if certainy is greater than or equal to the thresh, photon is included)
        Arguments ln and nstate are the expected number of tcspc bins and states
        respetively, used as not all bursts have nanotimes covering all tcspc bins
        or include all states.

        Parameters
        ----------
        nanos : np.ndarray[np.uint16]
            Nanotimes of burst.
        states : np.ndarray[np.uint8]
            *Viterbi* assigned state of burst.
        discr : np.ndarray[np.float64]
            Photons x states 2D array of likelihood for each state and each photon.
        thresh : float
            Threshold of discr to include photon in histogram.
        nstate : int
            Number of states.
        ln : int
            Number of TCSPC bins.
        nanosub : int
            Amount to subtract from nanos to set threshold

        Returns
        -------
        np.ndarray[np.int64]
            2D histogram, organized (state, tcspc bin).

        """
        
        mask = thresh <= discr
        return np.array([np.bincount(nanos[mask&(states==i)]-nanosub, minlength=ln) for i in range(nstate)])
    
    
    def get_nanohist_thresh_gamma(nanos:np.ndarray[np.int64], states:np.ndarray[np.int8],
                                  discr:np.ndarray[np.float64], thresh:float, 
                                  nstate:int, ln:int, nanosub:int)->np.ndarray[np.int64]:
        """
        Generate the 2D histogram of states x tcspc bin, given the photon nanotimes
        in ``nanos`` and the *Viterbi* state assignment in ``states``, while filtering
        out photons whose certainty of verterbi assigments 
        (should either be posterior probability or path loglikelihood) is less
        than ``thresh`` (if certainy is greater than or equal to the thresh, photon is included)
        Arguments ln and nstate are the expected number of tcspc bins and states
        respetively, used as not all bursts have nanotimes covering all tcspc bins
        or include all states.

        Parameters
        ----------
        nanos : np.ndarray[np.uint16]
            Nanotimes of burst.
        states : np.ndarray[np.uint8]
            *Viterbi* assigned state of burst.
        discr : np.ndarray[np.float64]
            Statistical likelihood of each state assignement of *Viterbi*.
        thresh : float
            Threshold of discr to include photon in histogram.
        nstate : int
            Number of states.
        ln : int
            Number of TCSPC bins.
        nanosub : int
            Amount to subtract from nanos to set threshold

        Returns
        -------
        np.ndarray[np.int64]
            2D histogram, organized (state, tcspc bin).

        """
        mask = thresh <= discr[range(discr.shape[0]),states]
        return np.array([np.bincount(nanos[mask&(states==i)]-nanosub, minlength=ln) for i in range(nstate)])


@fnumba.jit('i8[:,:](u1[:],i8)', boundscheck=True)
def count_trans(statepath:np.ndarray[np.uint8], nstate:np.int64)->np.ndarray[np.int64]:
    """
    Count the number of transitions from one state to another in a satepath.
    Counts the number of times each [statea, stateb] pair occurs in the statepath
    array.

    Parameters
    ----------
    statepath : np.ndarray[np.uint8]
        Statepath of indexes.
    nstate : np.int64
        Number of states in statepath, must be at least max(statepath) + 1, but
        if larger, the return value will be an array with 0's along the edges.

    Returns
    -------
    trans_map : np.ndarray[np.int64]
        2D array counting the number of times [start, stop] occurs in elements
        of statepath in succesive order.

    """
    trans_map = np.zeros((nstate, nstate), dtype=np.int64)
    for tloc in zip(statepath[:-1], statepath[1:]):
        trans_map[tloc] += 1
    return trans_map
        

    
class _ModelGetter:
    """
    Internal class, used to get model of specific number of states from a 
    sequence or dictionary of models.
    """
    def __init__(self, models:None|Sequence[hm.h2mm_model]|dict[int,hm.h2mm_model]):
        if models is None:
            self.models = None
        else:
            models = {m.nstate:m for m in models} if isinstance(models, Sequence) else models
            if isinstance(models, dict):
                self.models = models
                self.nextmodel = None
            else:
                self.models = iter(models)
                self.nextmodel = next(self.models)
    
    def get(self, i:int)->None|hm.h2mm_model:
        """
        Get model with i states, if not specified, return None
        """
        if self.models is None:
            return None
        if self.nextmodel is not None:
            cont = True
            model = None
            while cont and self.nextmodel.nstate < i:
                try:
                    self.nextmodel = next(self.models)
                except StopIteration:
                    cont = False
                if cont and self.nextmodel.nstate == i:
                    model = self.nextmodel
            return model
        else:
            return self.models.get(i, None)


def _conv_array(array:Sequence[float], thresh:float)->int:
    """
    Internal function for determining if and where an array of statistical
    discriminators has converged, assuming increasing numbers of states

    Parameters
    ----------
    array : Sequence[float]
        Sequence of statistical discriminators.
    thresh : float
        Maximum delta allowed to consider converted.

    Returns
    -------
    int
        Index of converged (ideal) model, -1 if not converged.

    """
    array = np.asarray(array)
    mask = (array - np.nanmin(array)) <= thresh
    if np.any(mask[:-1]):
        return np.argwhere(mask)[0,0]
    return -1


def _calc_stat_disc_array(origin:PhotonDataS, statepaths:Sequence[Param], stat:str, **kwargs)->np.ndarray[np.float64]:
    """Compute statistical discriminator stat for a given **Sequence** of statepaths"""
    return np.array([getattr(sp.tp, stat)(sp, origin, **kwargs) for sp in statepaths])


def calc_BIC(origin:PhotonDataS, statepaths:Sequence[Param], **kwargs)->np.ndarray[np.float64]:
    """
    Compute Bayes Information Criterion for sequence 
    (presumably from same optimization) of statepaths

    Parameters
    ----------
    origin : PhotonDataS
        Source of data (photons) for computation.
    statepaths : Sequence[Param]
        Sequence of :class:`StatePathBase` based :class:`Param` defining h2mm models.

    Returns
    -------
    np.ndarray[np.float64]
        BICs of each model in statpaths.

    """
    return _calc_stat_disc_array(origin, statepaths, 'BIC', **kwargs)


def calc_BICph(origin:PhotonDataS, statepaths:Sequence[Param], **kwargs)->np.ndarray[np.float64]:
    """
    Compute Bayes Information Criterion per photon for sequence 
    (presumably from same optimization) of statepaths

    Parameters
    ----------
    origin : PhotonDataS
        Source of data (photons) for computation.
    statepaths : Sequence[Param]
        Sequence of :class:`StatePathBase` based :class:`Param` defining h2mm models.

    Returns
    -------
    np.ndarray[np.float64]
        BIC per photon of each model in statpaths.

    """
    return _calc_stat_disc_array(origin, statepaths, 'BICph', **kwargs)


def calc_BICp(origin:PhotonDataS, statepaths:Sequence[Param])->np.ndarray[np.float64]:
    """
    Compute modified Bayes Information Criterion per photon for sequence 
    (presumably from same optimization) of statepaths

    Parameters
    ----------
    origin : PhotonDataS
        Source of data (photons) for computation.
    statepaths : Sequence[Param]
        Sequence of :class:`StatePathBase` based :class:`Param` defining h2mm models.

    Returns
    -------
    np.ndarray[np.float64]
        modified BIC per photon of each model in statpaths.

    """
    arrays = statepaths[0].tp._sort_photons(origin, statepaths[0])
    nphot = sum(i.size for i in arrays['indexes'].ravel())
    out = np.array([sp.tp.BIC(sp, origin)/(nphot - sp.params['model'].k) for sp in statepaths])
    return out - np.min(out)


def calc_ICL(origin:PhotonDataS, statepaths:Sequence[Param], **kwargs)->np.ndarray[np.float64]:
    """
    Compute Integrated Complete Likelihood for sequence 
    (presumably from same optimization) of statepaths

    Parameters
    ----------
    origin : PhotonDataS
        Source of data (photons) for computation.
    statepaths : Sequence[Param]
        Sequence of :class:`StatePathBase` based :class:`Param` defining h2mm models.

    Returns
    -------
    np.ndarray[np.float64]
        ICLs of each model in statpaths.

    """
    return _calc_stat_disc_array(origin, statepaths, 'ICL', **kwargs)


def calc_ICLph(origin:PhotonDataS, statepaths:Sequence[Param], **kwargs)->np.ndarray[np.float64]:
    """
    Compute Integrated Complete Likelihood per photon for sequence 
    (presumably from same optimization) of statepaths

    Parameters
    ----------
    origin : PhotonDataS
        Source of data (photons) for computation.
    statepaths : Sequence[Param]
        Sequence of :class:`StatePathBase` based :class:`Param` defining h2mm models.

    Returns
    -------
    np.ndarray[np.float64]
        ICL per photon of each model in statpaths.

    """
    return _calc_stat_disc_array(origin, statepaths, 'ICLph', **kwargs)


def calc_pathBIC(origin:PhotonDataS, statepaths:Sequence[Param], **kwargs)->np.ndarray[np.float64]:
    """
    Compute path Bayes Information Criterion for sequence 
    (presumably from same optimization) of statepaths

    Parameters
    ----------
    origin : PhotonDataS
        Source of data (photons) for computation.
    statepaths : Sequence[Param]
        Sequence of :class:`StatePathBase` based :class:`Param` defining h2mm models.

    Returns
    -------
    np.ndarray[np.float64]
        path BICs of each model in statpaths.

    """
    return _calc_stat_disc_array(origin, statepaths, 'pathBIC', **kwargs)


def calc_pathBICph(origin:PhotonDataS, statepaths:Sequence[Param], **kwargs)->np.ndarray[np.float64]:
    """
    Compute path Bayes Information Criterion per photon for sequence 
    (presumably from same optimization) of statepaths

    Parameters
    ----------
    origin : PhotonDataS
        Source of data (photons) for computation.
    statepaths : Sequence[Param]
        Sequence of :class:`StatePathBase` based :class:`Param` defining h2mm models.

    Returns
    -------
    np.ndarray[np.float64]
        path BIC per photon of each model in statpaths.

    """
    return _calc_stat_disc_array(origin, statepaths, 'pathBICph', **kwargs)



def conv_BIC(models:Sequence[hm.h2mm_model], origin:PhotonDataS, params:Sequence[Param], thresh:float=None)->int:
    """
    Determine ideal model in sequence of optimizations based on Bayes Information Criterion.

    Parameters
    ----------
    models : Sequence[hm.h2mm_model]
        Optimized models, in ascending number of states.
    origin : PhotonDataS
        Data on which optimizations were performed.
    params : Sequence[Param]
        StatePath Parameters of each optimized model.
    thresh : float, optional
        Minimum difference to accept model as ideal. If None, last model must
        have larger BIC than ideal (equivalent to thresh = 0.0)
        The default is None.

    Returns
    -------
    int
        Index of ideal model, if no ideal model found return -1.

    """
    thresh = 0.005 if thresh is None else thresh
    return _conv_array([m.bic for m in models], thresh)


def conv_BICph(models:Sequence[hm.h2mm_model], origin:PhotonDataS, params:Sequence[Param], thresh:float=None)->int:
    """
    Determine ideal model in sequence of optimizations based on 
    Bayes Information Criterion *per photon*.

    Parameters
    ----------
    models : Sequence[hm.h2mm_model]
        Optimized models, in ascending number of states.
    origin : PhotonDataS
        Data on which optimizations were performed.
    params : Sequence[Param]
        StatePath Parameters of each optimized model.
    thresh : float, optional
        Minimum difference to accept model as ideal. If None, last model must
        have larger BIC than ideal (equivalent to thresh = 0.0)
        The default is None.

    Returns
    -------
    int
        Index of ideal model, if no ideal model found return -1.

    """
    thresh = 0.005 if thresh is None else thresh
    return _conv_array([m.bic/m.nphot for m in models], thresh)


def conv_BICp(models:Sequence[hm.h2mm_model], origin:PhotonDataS, params:Sequence[Param], thresh:float=None)->int:
    """
    Determine ideal model in sequence of optimizations based on 
    modified Bayes Information Criterion.

    Parameters
    ----------
    models : Sequence[hm.h2mm_model]
        Optimized models, in ascending number of states.
    origin : PhotonDataS
        Data on which optimizations were performed.
    params : Sequence[Param]
        StatePath Parameters of each optimized model.
    thresh : float, optional
        Minimum difference to accept model as ideal. If None, last model must
        have larger BIC than ideal (equivalent to thresh = 0.0)
        The default is None.

    Returns
    -------
    int
        Index of ideal model, if no ideal model found return -1.

    """
    thresh = 0.005 if thresh is None else thresh
    bic = np.array([m.bic / (m.nphot - m.k) for m in models])
    bic -= bic.min()
    mask = bic < thresh
    if np.any(mask[:-1]):
        return np.argwhere(mask)[0,0]
    return -1


def conv_ICL(models:Sequence[hm.h2mm_model], origin:PhotonDataS, params:Sequence[Param], thresh:float=None, **kwargs)->int:
    """
    Determine ideal model in sequence of optimizations based on Integrated Complete Likelihood.

    Parameters
    ----------
    models : Sequence[hm.h2mm_model]
        Optimized models, in ascending number of states.
    origin : PhotonDataS
        Data on which optimizations were performed.
    params : Sequence[Param]
        StatePath Parameters of each optimized model.
    thresh : float, optional
        Minimum difference to accept model as ideal. If None, last model must
        have larger BIC than ideal (equivalent to thresh = 0.0)
        The default is None.

    Returns
    -------
    int
        Index of ideal model, if no ideal model found return -1.

    """
    thresh = 0.005 if thresh is None else thresh
    return _conv_array([param.ICL(origin, **kwargs) for param in params], thresh)


def conv_ICLph(models:Sequence[hm.h2mm_model], origin:PhotonDataS, params:Sequence[Param], thresh:float=None, **kwargs)->int:
    """
    Determine ideal model in sequence of optimizations based on 
    Integrated Complete Likelihood *per photon*.

    Parameters
    ----------
    models : Sequence[hm.h2mm_model]
        Optimized models, in ascending number of states.
    origin : PhotonDataS
        Data on which optimizations were performed.
    params : Sequence[Param]
        StatePath Parameters of each optimized model.
    thresh : float, optional
        Minimum difference to accept model as ideal. If None, last model must
        have larger BIC than ideal (equivalent to thresh = 0.0)
        The default is None.

    Returns
    -------
    int
        Index of ideal model, if no ideal model found return -1.

    """
    thresh = 0.005 if thresh is None else thresh
    return _conv_array([param.ICLph(origin, **kwargs) for param in params], thresh)


def conv_pathBIC(models:Sequence[hm.h2mm_model], origin:PhotonDataS, params:Sequence[Param], thresh:float=None, **kwargs)->int:
    """
    Determine ideal model in sequence of optimizations based on 
    path Bayes Information Criterion.

    Parameters
    ----------
    models : Sequence[hm.h2mm_model]
        Optimized models, in ascending number of states.
    origin : PhotonDataS
        Data on which optimizations were performed.
    params : Sequence[Param]
        StatePath Parameters of each optimized model.
    thresh : float, optional
        Minimum difference to accept model as ideal. If None, last model must
        have larger BIC than ideal (equivalent to thresh = 0.0)
        The default is None.

    Returns
    -------
    int
        Index of ideal model, if no ideal model found return -1.

    """
    thresh = 0.005 if thresh is None else thresh
    return _conv_array([param.pathBIC(origin, **kwargs) for param in params], thresh)


def conv_pathBICph(models:Sequence[hm.h2mm_model], origin:PhotonDataS, params:Sequence[Param], thresh:float=None, **kwargs)->int:
    """
    Determine ideal model in sequence of optimizations based on 
    path Bayes Information Criterion *per photon*.

    Parameters
    ----------
    models : Sequence[hm.h2mm_model]
        Optimized models, in ascending number of states.
    origin : PhotonDataS
        Data on which optimizations were performed.
    params : Sequence[Param]
        StatePath Parameters of each optimized model.
    thresh : float, optional
        Minimum difference to accept model as ideal. If None, last model must
        have larger BIC than ideal (equivalent to thresh = 0.0)
        The default is None.

    Returns
    -------
    int
        Index of ideal model, if no ideal model found return -1.

    """
    thresh = 0.005 if thresh is None else thresh
    return _conv_array([param.pathBICph(origin, **kwargs) for param in params], thresh)


class StatePathBase(Table):
    """
    Base class for tables storing *Viterbi* path of |H2MM| model. 
    
    Required param_defs:
        1. ParamDef('model', TV_H2MMModel),
        2. ParamDef('streams', TV_tuple(typedefs=TV_PhSel))
    
    Required parent_defs (automatically included):
        1. ColumnDef('bursts', BasePhotonColumn, is_base=True)
    
    Required columns
    
    Non-reordered array columns (all have rows of arrays of size of burst):
        1. indexpath indexes submitted to h2mm
        2. timepath times submitted to h2mm
        3. statepath simple output of most-likely states **set in init columns, store all**
        4. scalepath verterbi posterior probability **set in init columns, store all**
        5. pathllpath path photon likelihood, based on *Viterbi* path, **should be store all**
        6. gammapath h2mm state likelihood **should be store all**

    Reordered array columns (include ph_sel argument) size matches ph_times of base:
        7. ph_index (phsel:PhSel):  index mapped to phsel
        8. ph_h2mmtime (phsel:PhSel): times mapped to phsel (for alex, arrays may not be monotonic)
        9. ph_state (phsel:PhSel): state of each photon mapped to phsel
        10. ph_scale (phsel:PhSel): viterbi posterior probabilit mapped to phsel
        11. ph_pathll (phsel:PhSel): *Viterbi* path loglikilihood
        12. ph_gamma (phsel:PhSel): h2mm state likelihood

    Processed columns:
        13. ll (): loglikelihood of burst, given model and all possible state paths.
        14. llph (): loglikelihood per photon included in |H2MM| of burst, 
            given model and all possible state paths.
        15. eff_state (): effetive state (if state not know, infer from dwell time)
            computed columns
        16. bstates (): bitwise truthtable of states present
        17. transcount (): number of transitions between states in burst (2D array)
        18. ntrans (): number of transitions in burst, single integer
        19. pathll (): *Viterbi* path loglikelihood of the burst 
        20. pathllph (): *Viterbi* path loglikelihood of the burst per photon
        
        
    """
    
    _model_column_funcs = ImDict({'ratio_raw':[(BasePhotonTable, '_get_model_ratio'), ],
                                  'ratio_bg':[(NphBG, '_get_model_ratio'), ],
                                  'ratio_c':[(Ratios, '_get_model_ratio_corr'), ],
                                  'anisotropy_raw':[(BasePhotonTable, '_get_model_anisotropy'), ],
                                  'anisotropy_bg':[(NphBG, '_get_model_anisotropy'), ],
                                  'anisotropy_c':[(Ratios, '_get_model_anisotropy_corr'), ]})

    @classmethod
    def _get_val_str(cls, value:Any, keylen:int)->str:
        if isinstance(value, hm.h2mm_model):
            vstr = (f'H2MM Model [\nprior{value.prior}' +
                    '\ntrans ' + cls._get_val_str(value.trans, 4) +
                    '\nobs ' + cls._get_val_str(value.obs, 2) + '\n]').split('\n')
            return '\n'.join(' '*keylen+f'  {ln}' if i else ln for i, ln in enumerate(vstr))
        return super()._get_val_str(value, keylen)

    @classmethod
    def get_phsel_span(cls, param:Param)->PhSel:
        """
        Get the :class:`smfbursts.ph_sel.PhSel` object that spans all streams used
        to compute the phsel. 

        Parameters
        ----------
        param : Param
            StatePath for which to derive the "span" of phsels.

        Returns
        -------
        PhSel
            :class:`PhSel` which includes all streams used in H2MM processing.

        """
        return phsel_union(*param.params['streams'])

    def __post_init__(self):
        """Adds citations"""
        add_citation('PirchiJPCB2016', purpose='core H2MM analysis method')
        add_citation('HarrisNatComms2022', purpose='H2MM software')
        if len(self.param.params['streams']) > 2:
            add_citation('HarrisNatComms2022', purpose='multi-parameter H2MM analysis (using more than 2 photon streams)')

    @classmethod
    def param_idx_to_det_map(cls, params:dict[str:Any], detdef:DetDef)->np.ndarray[np.uint8]:
        """
        **Subclasses must implement this method**
        
        Returns 1D numpy array that should map 
        h2mm index to detector index- ie ``idxmap[index] = detector``
        where detector should match the ph_dets array
        
        Parameters
        ----------
        params : dict[str:Any]
            param dictionary definition for class, may omit keys not requied to
            compute the idx_to_det_map.
        detdef : DetDef
            :class:`DetDef` of data for which param is expected to be based.
            Needed to determing dets
        
        Returns
        -------
        np.ndaray[np.uint8]
            mapping of H2MM idx to det based on detdef.
        """
        raise NotImplementedError("subclasses must implement this method")

    # @abstractmethod
    @classmethod
    def _sort_photons(cls, origin:PhotonDataS, statepath:Param=None, **kwargs)->dict[str:np.ndarray[np.ndarray]]:
        """
        Implemented per subclass. Should return dictionary with (at minimum) keys
        'indexes' and 'times'
        That are given to hm.h2mm_model.evaluate(indexes, times) to evaluate
        and/or optimize a model given the data in statpath or kwargs
        """
        raise NotImplementedError("sublcasses must implement this method")

    @property
    def phsel_span(self)->PhSel:
        """
        The :class:`smfbursts.ph_sel.PhSel` object that includes all streams used
        in computing the :class:`H2MM_C.h2mm_model` used to determine if a given
        stream is included or excluded from |H2MM| processing
        """
        return self.get_phsel_span(self.param)

    @classmethod
    def get_ndet(cls, param:Param)->int:
        """
        The number detector indexes for a :class:`smfbursts.datamodel.tables.Param` of the given subclass.
        **Implemented per sublcass.**
        
        .. note::
            
            Should be based on parameters in param other than "model" so that
            it can be used to validate the model in validate_param

        Parameters
        ----------
        param : Param
            :class:`smfbursts.datamodel.tables.Param` of type matching class for which to determine
            the number of detector indices.

        Returns
        -------
        int
            Number of indeces in :class:`H2MM_C.h2mm_model` of the input 
            :class:`H2MM_C.h2mm_model`.

        """
        raise NotImplementedError("Subclasses must implement this method")
        
    @paramproperty
    def nstate(cls, param:Param)->int:
        """Number of states in model"""
        return param.params['model'].nstate

    @paramproperty
    def ndet(cls, param:Param)->int:
        """Number of states in model"""
        return param.params['model'].ndet

    def _get_indexpath(self)->np.ndarray[np.ndarray[np.uint8]]:
        """Getter function for indexpath column, gets indexes handed to :meth:`hm.h2mm_model.evaluate`"""
        return self._sort_photons(self.origin, statepath=self.param)['indexes']

    def _get_timepath(self)->np.ndarray[np.ndarray[np.int64]]:
        """Getter function for timepath column, gets times handed to :meth:`hm.h2mm_model.evaluate`"""
        return self._sort_photons(self.origin, statepath=self.param)['times']

    def _get_detpath(self)->np.ndarray[np.ndarray[np.uint8]]:
        """Getter function for detpath column, gets photon detectors, **not mapped** used in evaluating h2mm model"""
        raise NotImplementedError("subclasses must implement this column directly")

    def _get_pathllpath(self)->np.ndarray[np.ndarray[np.float64]]:
        """
        Getter function for pathllpath column. 
        Gets loglikihood of state assigned by viterbi per photon
        """
        ll = _h2mm_path_loglik(self.param.params['model'], self['indexpath'], self['timepath'], 
                            self['statepath'])
        return ll
    
    def _get_pathll(self)->np.ndarray[np.float64]:
        """Getter function for pathll, the path-loglikihood per burst"""
        return np.fromiter((np.sum(ll) for ll in self['pathllpath']), np.dtype('<f8'), count=self.size)
    
    def _get_pathllph(self)->np.ndarray[np.float64]:
        """Getter function for pathllph, the path-loglikihood per photon per burst"""
        return np.fromiter((np.sum(ll)/ll.size for ll in self['pathllpath']), np.dtype('<f8'), count=self.size)

    def _get_gammapath(self)->np.ndarray[np.ndarray[np.float64]]:
        """Getter function ofr gammapath column. Gets likelihood of each photon x state"""
        gamma = _h2mm_evaluate_gamma(self.param.params['model'], self['indexpath'], self['timepath'])
        return gamma

    def phsel_select(self, phsel:PhSel, col:str, fill:Any, dtype:np.dtype)->np.ndarray[np.object_]:
        r"""
        Subclasses must implement this method. This method should implement the
        mapping of the inputs to H2MM evaluation to the "unprocessed" versions.

        Parameters
        ----------
        phsel : PhSel
            A phsel object defining the output streams to return.
        col : str
            Name of column being returned.
        fill : Any
            Value to fill any photons that are in phsel but outside of phsel_span.
        dtype : np.dtype
            Data-type of output array.

        Raises
        ------
        NotImplementedError
            Subclass does not allow use of this function, usually indicates
            abstract class.

        Returns
        -------
        np.ndarray[np.object\_]
            If implemented should return object array of column maped to phsel

        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_ph_index(self, phsel:PhSel)->np.ndarray[np.ndarray[np.int8]]:
        """Getter function for ph_index column, ph_indexes masked and reordered by ph_sel."""
        return self.phsel_select(phsel, 'indexpath', -1, np.int8)

    def _get_ph_h2mmtime(self, phsel:PhSel)->np.ndarray[np.ndarray[np.int8]]:
        """
        Getter function for ph_h2mmtime column, times used in h2mm calculation, 
        but masked and reorderd by phsel.
        """
        return self.phsel_select(phsel, 'timepath', -1, np.int64)

    def _get_ph_state(self, phsel:PhSel)->np.ndarray[np.ndarray[np.int8]]:
        """
        Getter function of ph_state column, viterbi state of each photon, 
        masked and reordered by phsel.
        """
        return self.phsel_select(phsel, 'statepath', -1, np.int8)

    def _get_ph_scale(self, phsel:PhSel)->np.ndarray[np.ndarray[np.float64]]:
        """
        Getter function for ph_scale column, viterbi posterior likelihood 
        masked and reordered by phsel.
        """
        return self.phsel_select(phsel, 'scalepath', -1.0, np.float64)

    def _get_ph_pathll(self, phsel:PhSel)->np.ndarray[np.ndarray[np.float64]]:
        """
        Getter function for ph_pathll column, loglikelihood of viterbi state 
        assigment of photon, masked and reordered by phsel.
        """
        return self.phsel_select(phsel, 'pathllpath', np.nan, np.float64)

    def _get_ph_gamma(self, phsel:PhSel)->np.ndarray[np.ndarray[np.float64]]:
        """
        Getter function for ph_gamma column, loglikelihood of each state by photon.
        Masked and reordered by phsel.
        """
        
        return self.phsel_select(phsel, 'gammapath', np.nan, np.float64)
    
    def _get_ll(self)->np.ndarray[np.float64]:
        """Getter for ll column (loglik of bursts)"""
        return _h2mm_evaluate(self.param.params['model'], self['indexpath'], self['timepath'], ll=True)[1]

    def _get_llph(self)->np.ndarray[np.float64]:
        """Getter for llph column (loglik per photon of bursts)"""
        return self['ll'] / self.origin.get_table(self.param.base_param)['nph_raw', self.phsel_span]

    def _get_bstates(self)->np.ndarray[np.int64]:
        """Getter function for bstates column, bitwise mask of states present in burst"""
        return np.array([np.bitwise_or.reduce(1<<states.astype(np.int64)) 
                         for states in self['statepath']])
    
    def _iter_dwellcount(self)->Iterator[np.ndarray[np.int64]]:
        """Iter function computes state count per burst"""
        nstate = self.param.params['model'].nstate
        for statepath in self.iter_column('statepath'):
            yield np.bincount(statepath, minlength=nstate)
    
    def _iter_transcount(self)->Iterator[np.ndarray[np.int64]]:
        """Iter function counts transitions between specific states per burst (2D array)"""
        nstate = self.param.params['model'].nstate
        for statepath in self.iter_column('statepath'):
            yield count_trans(statepath, nstate)
    
    def _iter_ntrans(self)->Iterator[int]:
        """Iter function for number of transitions"""
        mask = ~np.eye(self.param.params['model'].nstate, dtype=np.bool_)
        for trans in self.iter_column('transcount'):
            yield np.sum(trans[mask])

    def _iter_eff_state(self, phsel:PhSel)->np.ndarray[np.int8]:
        """Iter function for eff_state column, state of each photon, infered if photon not in original phsel"""
        for mask, states, times in zip(self.base_table.iter_column('ph_mask', phsel),
                                       self.iter_column('ph_state', phsel_all),
                                       self.base_table.iter_column('ph_times', phsel_all)):
            yield infer_state(states, times)[mask]

    def _iter_nanohist_state(self, phsel:PhSel, thresh:float, discr:Literal['ph_gamma', 'ph_scale', 'ph_pathll'], full:bool())->np.ndarray[np.int64]:
        """Iter function ofr nanohist_state"""
        if not phsel.positive:
            phsel = phsel.render_positive(self.origin.detdef, convert_all=False)
        if full:
            mn = 0
            ln = np.max(self.origin.setup.tcspc_num_bins)
        else:
            elements = phsel.ex.elements if phsel.ex.kind else (i for i in range(self.detdef.ex) if i not in phsel.ex.elements)
            ex_ranges = np.concatenate([self.origin.setup.ex_ranges[i] for i in elements])
            mn, mx = np.min(ex_ranges), np.max(ex_ranges)
            ln = mx - mn
        nstate = self.param.params['model'].nstate
        func = get_nanohist_thresh_gamma if discr == 'ph_gamma' else get_nanohist_thresh
        for nanos, states, post in zip(self.parents['bursts'].iter_column('ph_nanos', phsel),
                                       self.iter_column('ph_state', phsel),
                                       self.iter_column(discr, phsel)):
            yield func(nanos, states, post, thresh, nstate, ln, mn)
    
    @classmethod
    def _regularizecolumn_nanohist_state(cls, source_param:Param, *args)->tuple[PhSel,float,Literal['ph_gamma', 'ph_scale', 'ph_pathll'], bool]:
        """Column regularization function for nanohist_state column"""
        if not args:
            raise TypeError("must specify at least phsel of nanohist_state")
        if len(args) > 4:
            raise TypeError("too many keys for nanohist_state, maximumn 4, PhSel and full, (full optional)")
        if not isinstance(args[0], PhSel):
            raise TypeError("must specify PhSel as first key in nanohist_state column")
        out = [args[0],]
        i = 1
        for tp, cast, default in ((Real, float, 0.0), (str, str, 'ph_gamma'), (bool, bool, False)):
            if i < len(args) and isinstance(args[i], tp):
                out.append(cast(args[i]))
                i += 1
            else:
                out.append(default)
        if i != len(args):
            raise ValueError("Unrecognized keys: {args")
        if out[2] not in ('ph_scale', 'ph_pathll', 'ph_gamma'):
            raise TypeError(f"dicr must be either 'ph_scale' or 'ph_pathll', not {out[2]}")
        if out[2] == 'ph_pathll':
            if out[1] > 0.0:
                raise ValueError("'ph_pathll' type thresholds must be less than 0")
        elif out[1] < 0.0 or out[1] > 1.0:
            raise ValueError(f"'{out[2]}' type thresholds must be greater than 0")
        return tuple(out)
    
    @tableproperty
    def nphot(cls, statepath:Param, origin:PhotonDataS, from_flag:bool=True)->int:
        if from_flag and statepath.has_flag(f'dataID{origin.dataID}', 'h2mm', 'nphot'):
            return statepath.get_flag(f'dataID{origin.dataID}', 'h2mm', 'nphot')
        nphot = sum(t.size for t in statepath.tp._sort_photons(origin, statepath=statepath)['times'])
        if from_flag:
            statepath.set_flag((f'dataID{origin.dataID}', 'h2mm', 'nphot'), nphot)
        return nphot

    @tableproperty
    def BIC(cls, statepath:Param, origin:PhotonDataS, as_array:bool=False, from_flag:bool=True)->float:
        """
        This is a tableproperty.
        
        Bayes Information Criterion of given StatePath (model + burst selection from origin)

        Parameters
        ----------
        statepath : Param
            StatePath which defines burst selection, |H2MM| model and photon selection
            params.
        origin : PhotonDataS
            Raw Data.
        as_array : bool, optional
            (Only for when origin is ``PhotonDataList``). If :code:`True` return
            BIC of each element in 
            The default is False.
        from_flag : bool, optional
            If True, check flags of statepath, if dataID of origin is present,
            and BIC present, return already computed, if False, force computation
            of BIC new. The default is False
        Returns
        -------
        float | np.ndarray[np.float64]
            BIC.

        """
        if as_array:
            return np.array([cls.BIC(statepath, d) for d in origin.datas])
        if statepath.has_flag(f'dataID{origin.dataID}', 'h2mm', 'bic'):
            return statepath.get_flag(f'dataID{origin.dataID}', 'h2mm', 'bic')
        tp = statepath.tp
        arrays = tp._sort_photons(origin, statepath=statepath)
        indexes, times = arrays['indexes'], arrays['times']
        model = _h2mm_evaluate(statepath.params['model'], indexes, times)
        if from_flag:
            statepath.update_flag((f'dataID{origin.dataID}', 'h2mm'), 
                                  {'loglik':model.loglik, 'nphot':model.nphot, 
                                           'k':model.k, 'bic':model.bic})
        return model.bic

    @tableproperty
    def BICph(cls, statepath:Param, origin:PhotonDataS, 
              as_array:bool=False, from_flag:bool=True)->float:
        """
        Bayes Information Criterion per photon of given StatePath 
        (model + burst selection from origin). That is the :math:`BIC / no.photons`

        Parameters
        ----------
        origin : PhotonDataS
            Raw Data.
        statepath : Param
            StatePath which defines burst selection, |H2MM| model and photon selection
            params.

        Returns
        -------
        float
            BIC per photon.

        """
        if as_array:
            return np.array([cls.BICph(statepath, d, from_flag=from_flag) for d in origin.datas])
        bic = statepath.BIC(origin, from_flag=from_flag)
        nphot = statepath.nphot(origin, from_flag=from_flag)
        return bic / nphot

    @staticmethod
    def BICp(statepaths:Sequence[Param], origin:PhotonDataS, from_flag:bool=True)->np.ndarray[np.float64]:
        """
        Compute a the Bayes Information Criterion per photon of the models defining
        the input sequence of :class:`StatePath` based Params .

        Parameters
        ----------
        statepaths : Sequence[Param]
            Sequence of :class:`StatePath` based Params, used to specify
            |H2MM| models..
        origin : PhotonDataS
            Data against which to compute the BIC per photon.

        Returns
        -------
        np.ndarray[np.float]
            1D array of BIC per photon for each model.

        """
        bic = np.array([statepath.BIC(origin, from_flag=from_flag) for statepath in statepaths])
        nphot = statepaths[0].nphot(origin, from_flag=from_flag)
        cf = np.array([nphot-statepath.params['model'].k for statepath in statepaths])
        bic -= np.nanmin(bic)
        return bic / cf

    @classmethod
    def _sumpath(cls, origin:PhotonDataS, statepath:Param, colname:str)->tuple[float, int]:
        """
        Get the sum of a given _path column given data and statpath param

        Parameters
        ----------
        origin : PhotonDataS
            Data on which path is based.
        statepath : Param
            statepath to get sum of colname.
        colname : str
            A _path column name.

        Returns
        -------
        tuple[float, int]
            path-sum, number of photons

        """
        tables = origin.get_table(statepath)
        if not isinstance(tables, Sequence): # generalizes to use either PhotonData or PhotonDataSet
            tables = (tables, )
        csum, nphot = 0.0, 0
        for table in tables:
            for path in table.iter_column(colname):
                if path.size < 2:
                    continue
                csum += path.sum()
                nphot += path.size
        return csum, nphot

    @tableproperty
    def ICL(cls, statepath:Param, origin:PhotonDataS, 
            as_array:bool=False, from_flag:bool=True)->float:
        """
        Get the Integrated Complete Likelihood (ICL) of the data in ``origin``
        specified by the model in ``statepath``. Note that this is a classmethod.

        Parameters
        ----------
        origin : PhotonDataS
            Data for which to compute the ICL.
        statepath : Param
            :class:`Param` defining the burst selection and model.

        Returns
        -------
        float
            ICL of data/model combination.

        """
        if as_array:
            return np.array([cls.ICL(statepath, d) for d in origin.datas])
        if from_flag and statepath.has_flag(f'dataID{origin.dataID}', 'h2mm', 'icl'):
            return statepath.get_flag(f'dataID{origin.dataID}', 'h2mm', 'icl')
        ll, nphot = cls._sumpath(origin, statepath, 'scalepath')
        icl = statepath['model'].k*np.log(nphot) - 2*ll
        if from_flag:
            statepath.update_flag((f'dataID{origin.dataID}', 'h2mm'), {'icl':icl, 'nphot':nphot})
        return icl

    @tableproperty
    def ICLph(cls, statepath:Param, origin:PhotonDataS, 
              as_array:bool=False, from_flag:bool=True)->float:
        """
        Get the Integrated Complete Likelihood (ICL) per photons of the data in ``origin``
        specified by the model in ``statepath``. Note that this is a classmethod.

        Parameters
        ----------
        origin : PhotonDataS
            Data for which to compute the ICL.
        statepath : Param
            :class:`Param` defining the burst selection and model.

        Returns
        -------
        float
            ICL/photon of data/model combination.

        """
        if as_array:
            return np.array([cls.ICLph(statepath, d) for d in origin.datas])
        if from_flag:
            if (flag := statepath.get_flag(f'dataID{origin.dataID}')) is not None:
                if 'h2mm' in flag and 'pathbic' in flag['h2mm']:
                    return flag['h2mm']['pathbic']
        arrays = statepath.tp._sort_photons(origin, statepath=statepath)
        _, _, ll, icl = _viterbi_path(statepath.params['model'], arrays['indexes'], arrays['times'])
        if from_flag:
            statepath.update_flag((f'dataID{origin.dataID}', 'h2mm', 'icl'), icl)
        return icl

    @tableproperty
    def pathBIC(cls, statepath:Param, origin:PhotonDataS, 
                as_array:bool=False, from_flag:bool=True)->float:
        """
        Get the BIC of likelihood of most-likely path of the data in ``origin``
        specified by the model in ``statepath``. Note that this is a classmethod.

        Parameters
        ----------
        origin : PhotonDataS
            Data for which to compute the ICL.
        statepath : Param
            :class:`Param` defining the burst selection and model.

        Returns
        -------
        float
            path BIC of data/model combination.

        """
        if as_array:
            return np.array([cls.pathBIC(statepath, d) for d in origin.datas])
        if from_flag:
            if (flag := statepath.get_flag(f'dataID{origin.dataID}')) is not None:
                if 'h2mm' in flag and 'pathbic' in flag['h2mm']:
                    return flag['h2mm']['pathbic']
        ll, nphot = cls._sumpath(origin, statepath, 'pathllpath')
        pathbic = statepath['model'].k*np.log(nphot) - 2*ll
        if from_flag:
            statepath.update_flag((f'dataID{origin.dataID}', 'h2mm'), 
                                  {'pathbic':pathbic, 'nphot':nphot})
        return pathbic

    @tableproperty
    def pathBICph(cls, statepath:Param, origin:PhotonDataS, 
                  as_array:bool=False, from_flag:bool=True)->float:
        """
        Get the BIC of likelihood of most-likely state-path per photon of the 
        data in ``origin`` specified by the model in ``statepath``. 
        Note that this is a classmethod.

        Parameters
        ----------
        origin : PhotonDataS
            Data for which to compute the ICL.
        statepath : Param
            :class:`Param` defining the burst selection and model.

        Returns
        -------
        float
            path BIC of data/model combination.

        """
        if as_array:
            return np.array([cls.pathBICph(statepath, d, from_flag=True) for d in origin.datas])
        pathbic = statepath.pathBIC(origin, from_flag=from_flag)
        nphot = statepath.nphot(origin, from_flag=from_flag)
        return pathbic / nphot    
    
    @parammethod(origin_as_kw=True)
    def model_streams(cls, statepath:Param, phsel:PhSel, origin:PhotonDataS=None, strict:bool=True)->np.ndarray[np.int64]:
        """
        Determine index(es) of ``phsel`` in the :class:`hm.h2mm_model` based on
        a given param definition.

        Parameters
        ----------
        statepath : Param
            StatePath Param to derive model from.
        phsel : PhSel
            phsel to extract model detector indexes.
        origin : PhotonDataS, optional
            Origin Data. The default is None.
        strict : bool, optional
            Whether to require phsel in phsel_span of statepath. 
            The default is True.
        
        Returns
        -------
        np.ndarray[np.int64]
            array of indexes- `statepath.model.obs[:,out]` will return array
            of only obs columns in phsel.

        """
        raise NotImplementedError("Subclasses must impelement this method")

    @classmethod
    def param_model(cls, statepath:Param, origin:PhotonDataS=None)->hm.h2mm_model:
        """
        Get the model from a given statepath of matching type.
        **Implemented per subclass**
        Implemented so that statpaths can define the model either in params or parents. 

        Parameters
        ----------
        statepath : Param
            Param from which to extract model.
        origin : PhotonDataS, optional
            Origin data. The default is None.

        Returns
        -------
        hm.h2mm_model
            Model of statepath

        """
        raise NotImplementedError("Subclasses must impelement this method")

    @parammethod(origin_as_kw=True)
    def param_streams(cls, statepath:Param, origin:PhotonDataS=None)->Sequence[PhSel]:
        """
        Get the sequence of phsels specifying each stream in obs.
        **Implemented per subclass**

        Parameters
        ----------
        statepath : Param
            StatePath base :class:`Param` from which to extract streams.
        origin : PhotonDataS, optional
            Origin Data. The default is None.

        Returns
        -------
        Sequence[PhSel]
            Sequence of phsel, specifying map of stream -> |H2MM| index.

        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @tableproperty
    def transrate(cls, statepath:Param, origin:PhotonDataS)->np.ndarray[np.float64]:
        """
        Table property.
        
        Compute the transition rates of the model in units of s.

        Parameters
        ----------
        statepath : Param
            Param specifying |H2MM| model
        origin : PhotonDataS
            Data used to optimize model, used to define clock rate.

        Returns
        -------
        np.ndarray[np.float64]
            Transition rates in seconds.

        """
        return statepath.params['model'].trans / origin.clk_p

    @classmethod
    def _get_model_nph_streams(cls, phsel:PhSel, statepath:Param, strict:bool=True, origin:PhotonDataS=None)->np.ndarray[np.float64]:
        """Retrieve array of [state, [det in phsel]] to compute nph of model"""
        used_streams = cls.model_streams(statepath, phsel, strict=strict, origin=origin)
        return cls.param_model(statepath, origin=origin).obs[:, used_streams]

    @classmethod
    def _get_model_nph(cls, col:Column, statepath:Param, strict:bool=True, origin:PhotonDataS=None)->np.ndarray[np.float64]:
        """col must be nph column, get likelihood per state of phsel of col, based on model"""
        return cls._get_model_nph_streams(col.keytup[0], statepath, strict=strict, origin=origin).sum(axis=1)

    @classmethod
    def _get_model_ratio(cls, col:Column, statepath:Param, strict:bool=True, origin:PhotonDataS=None)->np.ndarray[np.float64]:
        """col must be ratio column, get ratio per state of col, based on model"""
        num = cls._get_model_nph_streams(col.keytup[0], statepath, strict=strict, origin=origin).sum(axis=1)
        dem = cls._get_model_nph_streams(col.keytup[1], statepath, strict=strict, origin=origin).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            rat = num / dem
        return rat

    @classmethod
    def _get_model_anisotropy(cls, col:Column, statepath:Param, strict:bool=True, origin:PhotonDataS=None)->np.ndarray[np.float64]:
        """col must be anisotropy column, get anisotropy per state of col based on model"""
        p = cls._get_model_nph_streams(col.keytup[0], statepath, strict=strict, origin=origin).sum(axis=1)
        s = cls._get_model_nph_streams(col.keytup[1], statepath, strict=strict, origin=origin).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ani = (p - s) / (p + 2*s)
        return ani

    @classmethod
    def _get_model_nph_corr_streams(cls, corr_mat:np.ndarray[np.float64], phsels:Sequence[PhSel], 
                                    statepath:Param,
                                    strict:bool=True, origin:PhotonDataS=None)->np.ndarray[np.float64]:
        """Specifically for nph_c type columns, get corrected nph values from model per stream"""
        streams = cls.param_streams(statepath, origin=origin)
        detdef = statepath.detdef
        ms = (cls.model_streams(statepath, stream, origin=origin, strict=strict) for stream in streams)
        omat = np.array([cls.param_model(statepath, origin=origin).obs[:,m].sum(axis=1) for m in ms])
        si = tuple(detdef.get_stream_ids(stream) for stream in streams)
        cmat = np.array([[corr_mat[ri, ci].sum()/ci.size for ci in si] for ri in si])
        cstreams =  cmat @ omat
        masks = (np.array([stream in phsel for stream in streams]) for phsel in phsels)
        return tuple(cstreams[mask,:].sum(axis=0) for mask in masks)

    @classmethod
    def _get_model_nph_corr(cls, col:Column, statepath:Param, strict:bool=True, origin:PhotonDataS=None)->np.ndarray[np.float64]:
        """col must be nph_c type column, get corrected ratios from column per state based on model"""
        return cls._get_model_nph_corr_streams(col.param.params['corr_mat'], col.keytup[:1], statepath, strict=strict)[0]

    @classmethod
    def _get_model_ratio_corr(cls, col:Column, statepath:Param, strict:bool=True, origin:PhotonDataS=None)->np.ndarray[np.float64]:
        """col must be ratio_c type column, get corrected ratios from column per state based on model"""
        num, dem = cls._get_model_nph_corr_streams(col.param.params['corr_mat'], 
                                                   col.keytup[:2], statepath, 
                                                   strict=strict)
        return num / dem

    @classmethod
    def _get_model_anisotropy_corr(cls, col:Column, statepath:Param, strict:bool=True, origin:PhotonDataS=None)->np.ndarray[np.float64]:
        """col must be anisotropy_c type column, get corrected ratios from column per state based on model"""
        p, s = cls._get_model_nph_corr_streams(col.param.params['corr_mat'], 
                                               col.keytup[:2], statepath, strict=strict)
        return (p - s) / (p + 2*s)

    @parammethod(origin_as_kw=True)
    def model_value(cls, statepath:Param, col:Column, origin:PhotonDataS=None, strict:bool=True)->np.ndarray[np.float64]:
        """
        A parammethod that retrives the expected value of column per state
        based on the model of statepath.
        

        Parameters
        ----------
        statepath : Param
            StatePath based param defining H2MM model.
        col : Column
            Column to compute expected values based on H2MM model.
        origin : PhotonDataS, optional
            Origin data. The default is None.
        strict : bool, optional
            Ensure value can be fully calculated from model 
            (ie phsel does not contain streams not used in model-streams). 
            The default is True.
        
        Raises
        ------
        ValueError
            One or more phsel of col outside of model-strams.

        Returns
        -------
        np.ndarray[np.float64]
            Per-state expected value of column.

        """
        funcs = cls._model_column_funcs.get(col.col)
        if funcs is None:
            raise ValueError(f"No conversion function for columns of {col.col} for class {cls.__name__}")
        for tp, func in funcs:
            if issubclass(col.param.tp, tp):
                return getattr(cls, func)(col, statepath, origin=origin, strict=strict)
        raise ValueError(f"No conversion function for columns of {col.col} derived from {col.param.tp.__name__} for class {cls.__name__}")

    @parammethod(origin_as_kw=True)
    def model_values(cls, statepath:Param, *args:Column, origin:PhotonDataS=None, strict:bool=True)->tuple[np.ndarray[np.float64],...]:
        """
        Get expected values of multiple columns of Table based on model.

        Parameters
        ----------
        *args : Column
            Columns to get expected values based on model.
        strict : bool, optional
            Whether or not to raise an error if any col uses streams outside of 
            model-streams. The default is True.

        Returns
        -------
        np.ndarray[np.float] ...
            Arrays of per-state expected values of each column based on model.

        """
        return tuple(cls.model_value(statepath, arg, origin=origin, strict=strict) for arg in args)

    @classmethod
    def _find_saved(cls, data:PhotonData, states:int|Sequence[int]=None)->tuple[Param,...]:
        """Locate and return list of any tables saved to disk with type matching cls"""
        if not data._group._creatable:
            return tuple()
        if isinstance(states, Integral):
            states = (states,)
        tables = list()
        for group in data._group._file.iter_nodes(data._group._group):
            if group._v_name.startswith('TABLE'):
                param = TypeValidator.read_any(group.param)
                if issubclass(param.tp, cls):
                    if states is None or param.params['model'].nstate in states:
                        tables.append(param)
        return sorted(tables, key=lambda t: t.params['model'].nstate)

    @classmethod
    def find_saved(cls, data:PhotonDataS, states:int|Sequence[int]=None)->tuple[Param,...]:
        """
        Retrieve all saved tables matching class and return as tuple of params

        Parameters
        ----------
        data : PhotonDataS
            DESCRIPTION.
        states : int | Sequence[int], optional
            If specified, restrict parmas returned to those with models matching states. 
            The default is None.

        Returns
        -------
        tuple[Param,...]
            Tuple of all matching Params.

        """
        if isinstance(data, PhotonDataList):
            data = data.datas[0]
        return cls._find_saved(data, states)

    @classmethod
    def load_saved_models(cls, data:PhotonDataS, states:int|Sequence[int]=None)->tuple[Param,...]:
        """
        Load saved models into memory, and return tuple of their Params.

        Parameters
        ----------
        data : PhotonDataS
            Data with linked HDF5 file from which to load models 
            (models saved in HDF5 file).
        states : int | Sequence[int], optional
            If specified, restrict parmas returned to those with models matching states. 
            The default is None.

        Returns
        -------
        tuple[Param,...]
            Tuple of all matching Params.

        """
        params = cls.find_saved(data, states)
        for param in params:
            data.get_table(param)
        return params


TV_str_nhstate = TV_str(isin=('ph_scale', 'ph_pathll', 'ph_gamma'))
h2mm_paramdefs = (ParamDef('model', TV_H2MMModel), ParamDef('streams', TV_tuple(typedefs=TV_PhSel)),)
h2mm_columndefs = (
    # results of h2mm algorithms
    ColumnDef('indexpath', tuple(), 0, 'user', get_func='_get_indexpath', 
              dtype=np.object_, typedef=np.dtype('<u1'), unit='index'),
    ColumnDef('detpath', tuple(), 0, 'user', get_func='_get_detpath', 
              dtype=np.object_, typedef=np.dtype('<u1'), unit='index'),
    ColumnDef('timepath', tuple(), 0, 'user', get_func='_get_timepath',
              dtype=np.object_, typedef=np.dtype('<i8')),
    ColumnDef('statepath', tuple(), 0, 'all', unit='state',
              dtype=np.object_, typedef=np.dtype('<u1')),
    ColumnDef('scalepath', tuple(), 0, 'all', unit='posterior likelihood',
              dtype=np.object_, typedef=np.dtype('<f8')),
    ColumnDef('pathllpath', tuple(), 0, 'user', get_func='_get_pathllpath',
              dtype=np.object_, typedef=np.dtype('<f8'), unit='likelihood'),
    ColumnDef('pathll', tuple(), 0, 'user', get_func='_get_pathll', 
              dtype=np.dtype('<f8'), unit='loglik'),
    ColumnDef('pathllph', tuple(), 0, 'user', get_func='_get_pathllph', 
              dtype=np.dtype('<f8'), unit='loglik ph^{-1}'),
    ColumnDef('gammapath', tuple(), 0, 'user', get_func='_get_gammapath', 
              dtype=np.object_, typedef=np.dtype('<f8'), unit='likelihood'),
    # PhSel mapped versions
    ColumnDef('ph_index', (PhSel, ), 0, 'never', get_func='_get_ph_index',
              dtype=np.object_, typedef=np.dtype('<i1')),
    ColumnDef('ph_h2mmtime', (PhSel, ), 0, 'never', get_func='_get_ph_h2mmtime',
              dtype=np.object_, typedef=np.dtype('<i8')),
    ColumnDef('ph_state', (PhSel, ), 0, 'user', get_func='_get_ph_state',
              dtype=np.object_, typedef=np.dtype('<i1')),
    ColumnDef('ph_scale', (PhSel, ), 0, 'user', get_func='_get_ph_scale',
              dtype=np.object_, typedef=np.dtype('<f8')),
    ColumnDef('ph_pathll', (PhSel, ), 0, 'user', get_func='_get_ph_pathll',
              dtype=np.object_, typedef=np.dtype('<f8')),
    ColumnDef('ph_gamma', (PhSel, ), 0, 'user', get_func='_get_ph_gamma', 
              dtype=np.object_, typedef=np.dtype('<f8')),
    ColumnDef('ll', tuple(), 0, 'user', get_func='_get_ll', dtype=np.dtype('<f8')),
    ColumnDef('llph', tuple(), 0, 'user', get_func='_get_llph', dtype=np.dtype('<f8')),
    # processed columns
    ColumnDef('bstates', tuple(), 0, 'user', get_func='_get_bstates', dtype=np.dtype('<i8')),
    ColumnDef('dwellcount', tuple(), 0, 'user', get_func='_iter_dwellcount', 
              get_derived=True, dtype=np.dtype('<i8'), ndim=2, unit='cnts'),
    ColumnDef('transcount', tuple(), 0, 'user', iter_func='_iter_transcount',
              get_derived=True, dtype=np.dtype('<i8'), ndim=3, unit='cnts'),
    ColumnDef('ntrans', tuple(), 0, 'user', iter_func='_iter_ntrans', 
              dtype=np.dtype('<i8'), unit='cnts'),
    ColumnDef('eff_state', (PhSel, ), 0, 'user', iter_func='_iter_eff_state',
              dtype=np.object_, typedef=np.dtype('<i1')),
    ColumnDef('nanohist_state', (PhSel, float, TV_str_nhstate, bool), 0, 'never',
              iter_func='_iter_nanohist_state', reg_func='_regularizecolumn_nanohist_state',
              get_derived=True, dtype=np.dtype('<i8'), ndim=3),
    )


def make_h2mm_columndefs(skip:str|Sequence[str]=None)->tuple[ColumnDef,...]:
    """
    Make tuple of columndefs for StatePathBase subclass.
    Defines the folloing columns:
    
        - indexpath
        - detpath, subclass should implement _get_detpath getter function
        - timepath
        - statepath
        - scalepath
        - pathllpath
        - pathll
        - pathllph
        - gamapath
        - ph_index
        - ph_h2mmtime
        - ph_state
        - ph_scale
        - ph_pathll
        - ph_gamma
        - bstates
        - eff_states

    Parameters
    ----------
    skip : str | Sequence[str], optional
        name(s) of :class:`ColumnDef` s to skip. The default is None.

    Returns
    -------
    tuple[ColumnDef,...]
        tuple of :class:`ColumnDef` for use in StatePathBase.column_defs.

    """
    skip = tuple() if skip is None else skip
    skip = (skip, ) if isinstance(skip, str) else skip
    return tuple(cdef for cdef in h2mm_columndefs if cdef.name not in skip)


def _validate_model_streams(streams:Sequence[PhSel], detdef:DetDef)->int:
    """
    Validator for StatePath classes where ndet == len(streams), 
    returns number of streams, and checks all streams valid for detdef
    """
    nstream = len(streams)
    if any(not stream.positive_all for stream in streams):
        raise ValueError("Negative streams not allowed in streams param")
    stream_ids = [detdef.get_stream_ids(stream) for stream in streams]
    if any(np.intersect1d(sa, sb).size for sa, sb in permutations(stream_ids, 2)):
        raise ValueError("Overlapping stream definitions not allow in streams param")
    if np.any(np.diff([stream[0] for stream in stream_ids]) < 1):
        raise ValueError("Streams in non-ascending order")
    return nstream


def _astype(dtype:np.dtype, arr:np.ndarray)->np.ndarray:
    """Convert array to specified type"""
    return arr.astype(dtype)

# div schemes:
# lt- use present lifetime as source of exponential divisor
# cum- use decay as source
# option: include/exclude irf
# option: number of divs

_CCritLit = Literal['BIC','BICp','BICph','ICL','ICLph','pathBIC','pathBICph']
_CCritFunc = Callable[[Sequence[hm.h2mm_model],PhotonDataS,Sequence[Param],float|None],int]
#: Options for convergence criterion of :meth:`StatePath.optimize_models`
CCrit = _CCritLit|_CCritFunc


class StatePath(StatePathBase, ChildPhotonTable):
    """
    Basic |H2MM| *Viterbi* path of each burst based on :class:`hm.h2mm_model`.
    the model is defined in ``model`` parameter, and the the maping of 
    photon stream to |H2MM| index is defined by the  ``streams`` parameter.
    Photon streams are ``smf.PhSel`` objects, the may specify a single 
    detector id or group multiple streams together. If a given detector id is
    not specified by any photon stream in ``streams`` it is ommitted from
    data given to the viterbi algorithm.
    
    The most common method of creating ``StatePath`` objects is 
    :meth:`StatePath.optimize_models` which 
    
    >>> statepaths = bhm.StatePath.optimize_models(data, bursts, max_states=8, to_state=4, conv_crit='pathBIC')
    
    Based on the above call, ``statepaths`` will contain at least 4 and up to 8 
    ``StatePath`` ``Param`` objects, from 1 state to the point when the convergence 
    criterion was met (pathBIC minimized), maximally 8 states.
    
    Params
    ------
        model : hm.h2mm_model
            The :class:`hm.h2mm_model` used in *Viterbi* processing.
        streams : tuple[PhSel, ...]
            tuple of :class:`PhSel` defining the indexes of photons in |H2MM| processing.
        
    Parents
    -------
        bursts : Param[BasePhotonTable]
            Usually a :class:smfbursts.datamodel.tables.Bursts` :class:`smfbursts.datamodel.tables.Param`, defining the time 
            ranges of each "burst" in |H2MM| processing.
    
    Columns
    -------
        indexpath : np.ndarray[np.uint8], ()
            Actual indexes used in |H2MM| processing. Each row is uint8 array.
        detpath : np.ndarray[np.uint8], ()
            Detector indexes of photons used in |H2MM| processing, not reassigned by streams.
            Each row is uint8 array.
        timepath : np.ndarray[np.int64], ()
            Actual times used in |H2MM| processing. Each row is int64 array.
        statepath : np.ndarray[np.uint8], ()
            Most likely states of each photon as processed in *Viterbi* algorithm.
            This is the direct output of *Viterbi*, no sub-selecting photons etc.
            Each row is 1d uint8 array
        scalepath : np.ndarray[np.ndarray[np.float64]], ()
            Posterior per photon of *Viterbi* processing. This is direct output of *Viterbi*.
            Each row is 1D float64 array.
        pathllpath : np.ndarray[np.float64], ()
            Log-likelihood of most likely state of each photon, assuming most
            likely statepath. This is direct output of *Viterbi*.
            Each row is 1D float64 array.
        gammapath : np.ndarray[np.float64], ()
            Gamma array, giving likelihood per-photon per-state.
            Each row is 2d float64 array, indexed [photon, state].
        ph_index : np.ndarray[np.int8] (phsel:PhSel, )
            Indexes (submitted to |H2MM| processing) mapped/maskes by phsel.
            Photons outside of model-stream recieve a value of -1
        ph_h2mmtime : np.ndaray[np.int64], (phsel:PhSel, )
            Times of photons used in |H2MM| processing mapped/masked by phsel. 
            Photons outside of model-streams receive value of -1. 
            Rows are 1d int64 arrays.
        ph_state : np.ndarray[np.int8], (phsel:PhSel, )
            *Viterbi* state of each photon, mapped/masked by phsel. 
            Photons outside of model-streams receive value of -1. 
            Rows are 1d int8 arrays.
        ph_scale : np.ndarray[np.float64], (phsel:PhSel, )
            Posterior likelihood of each phton, mapped/masked by phsel.
            Photons outside of model-streams receive value of nan.
            Rows are 1d float64 arrays.
        ph_pathll : np.ndarray[np.float64], (phsel:PhSel, )
            log-likihood of state-assignment of photon for *Viterbi* path, mapped/masked by phsel.
            Photons outside of model-streams receive value of nan.
            Rows are 1d float64 arrays
        ph_gamma : np.ndarray[np.float64], (phsel:PhSel, )
            Gamma array, giving likelihood per-photon per-state mapped/masked by phsel.
            Each row is 2d float64 array, indexed [photon, state].
        ll : float, ()
            loglikelihood of each burst, over all statepaths.
        llph : float ()
            The loglikelihood per photon over all statepaths of each burst.
        pathll : float, ()
            loglikelihood of each burst, of most likely statpath.
        pathllph : float ()
            The loglikelihood per photon of most likely statpath of each burst.
        eff_state : np.ndarray[np.int8], (phsel:PhSel, )
            "Effective" state of each photon in burst maped/masked by phsel. 
            If a photon is in a stream not present in model streams, then infer state 
            by nearest photon that is in streams.
            This is essentially ph_state with -1s replaced with an infered state.
            Rows are 1d int8 arrays.
        bstates : int, ()
            Bitcode indicating which states present in burst. If given bit position
            is present, then that state is present in burst, ie if states 0 and 2 are
            in a given burst, the the value is 0b00000101 = 5. Rows are int.
        transcount : np.ndarray[np.int64], ()
            Number of transisions from state to state in a given burst.
            2D square rows, size of nstate.
        ntrans : int, ()
            Number of transitions in burst, sum of off-diagonal of transcount.
        nanohist_state : np.ndarray[np.int64], (phsel:PhSel, thresh:float, discr:Literal['ph_gamma', 'ph_state', 'ph_pathll'], full:bool)
            Histogram of state x nanotime of photons in a burst, with photons 
            filtered by thresh and discr, only photons with a likelihood of state
            assignment greater than or equal to thresh according to the dicr
            (either ph_gamma, ph_scale or ph_pathll) are included in the histogram. If full, the
            return histogram using TCSPC raw bins, if full=False, then trim to excitation range.
    
    """
    parent_defs = (ParentDef('bursts', BasePhotonTable, is_base=True), ) #: :meta private:
    param_defs = h2mm_paramdefs #: :meta private:
    column_defs = h2mm_columndefs #: :meta private:
    _conv_funcs = {'BIC':conv_BIC, 'BICp':conv_BICp, 'BICph':conv_BICph,
                   'ICL':conv_ICL, 'ICLph':conv_ICLph,
                   'pathBIC':conv_pathBIC, 'pathBICph':conv_pathBICph}
    #: For subclasses, any additional values (in order of return after indexes and times)
    #: to store in cache
    _sort_store = tuple()

    def __init_columns__(self):
        arrays  = self._sort_photons(self.origin, statepath=self.param)
        indexes, times = arrays['indexes'], arrays['times']
        path, scale, _, _ = _viterbi_path(self.param.params['model'], indexes, times)
        self._add_column('statepath',  tuple(), path)
        self._add_column('scalepath', tuple(), scale)
        if self.origin.save_memory:
            self.record_column('ll')
        self._temp_cache = arrays if self.origin.save_memory else dict()

    def _save_memory_switch(self, switch:bool):
        "Switch storing coputed index/times pairs in cache"
        if not switch:
            self._temp_cache = dict()
        
    @classmethod
    def get_ndet(cls, param:Param)->int:
        """
        From :class:`Param` based on StatePath, get the number of indexes assigned
        to photon streams.

        Parameters
        ----------
        param : Param
            :class:`Param:` based on Statepath defining number of indexes in H2MM processing.

        Returns
        -------
        int
            Number of indexes used in H2MM processing defined by param.

        """
        return len(param.params['streams'])

    @classmethod
    def param_ndet(cls, param:dict)->int:
        """
        Get number of indexes for H2MM processing defined by param dictionary.
        This dictionary need only define the parts of the param in a StatePath
        Param needed to determine the number of indexes. Other keys may be omitted.

        Parameters
        ----------
        param : dict
            params dict to be used to create StatePath :class:`Param`, unnecessary
            keys may be ommited.

        Returns
        -------
        int
            Number of indexes in H2MM processing as defined by param.

        """
        return len(param['streams'])

    @classmethod
    def validate_param(cls, param:Param)->None:
        """Validate a StatePath :class:`Param` :meta private:"""
        ndet = _validate_model_streams(param.params['streams'], param.detdef)
        if ndet < 2:
            raise ValueError("must specify at least 2 streams")
        if (mndet:=param.params['model'].ndet) !=ndet:
            raise ValueError(f"Mismatched model to number of dets, defined {ndet} detectors but model has {mndet}")

    @classmethod
    def param_preprocess(cls, params:Sequence[tuple[str,Any]]|tupledict, parents:dict[str,Param])->tuple[dict,dict]:
        """Preprocess auto-filling detdef and sorting model :meta private:"""
        params = as_paramdict(params, tuple(pdef.name for pdef in cls.param_defs))
        parents = as_paramdict(parents, tuple(pdef.name for pdef in cls.parent_defs))
        if not issubclass(parents['bursts'].tp, BasePhotonTable):
            parents['bursts'] = parents['bursts'].base_param
        detdef = parents['bursts'].detdef
        params, sort = cls.param_sort_process(params, detdef)
        if np.any(np.diff(sort) != 1):
            model = params['model']
            params['model'] = hm.h2mm_model(model.prior, model.trans, model.obs[:,sort])
        return params, parents

    @classmethod
    def param_sort_process(cls, params:dict[str:Any], detdef:DetDef)->tuple[dict[str:Any],np.ndarray[np.int64]]:
        """
        In a params dictonary, sort the streams so that streams are in ascending
        order based on DetDef
        
        Parameters
        ----------
        params : dict[str:Any]
            params dictionary to be used in StatePath based :class:`Param`.
        detdef : DetDef
            :class:`DetDef` of expected :class:`Param`.

        Returns
        -------
        params : dict[str,Any]
            Sorted params dictoinary.
        sort : np.ndarray[np.int64]
            Re-sorting array, value is original index, position is index for destination.
            Therefore new = old[sort]

        """
        if params.get('streams',None) is None:
            params['streams'] = tuple(detdef.stream_ids_to_PhSel(i, convert_all=True) 
                                      for i in range(detdef.size))
            sort = np.arange(detdef.size)
        else:
            params['streams'], sort = sort_phsels(params['streams'], 
                                                  detdef=detdef, return_index=True)
        return params, sort
    
    @classmethod
    def param_idx_to_det_map(cls, params:dict[str:Any], detdef:DetDef)->np.ndarray[np.uint8]:
        """
        Returns 1D numpy array that should map 
        h2mm index to detector index- ie ``idxmap[index] = detector``
        where detector should match the ph_dets array
        
        Parameters
        ----------
        params : dict[str:Any]
            param dictionary definition for class, may omit keys not requied to
            compute the idx_to_det_map.
        detdef : DetDef
            :class:`DetDef` of data for which param is expected to be based.
            Needed to determing dets
        
        Returns
        -------
        np.ndaray[np.uint8]
            mapping of H2MM idx to det based on detdef.
        """
        return np.array([detdef.get_stream_ids(stream)[0] for stream in params['streams']], dtype='<u1')

    @classmethod
    def _sort_photons_func(cls, origin:PhotonData, bursts:Param, streams:Sequence[PhSel]
                           )->dict[str:np.ndarray[np.ndarray]]:
        """Inner classmethod for performs sorting of sorts photon times to return dictionary"""
        indexes, times = sort_indexes_times(origin.get_table(bursts), streams)
        return dict(indexes=indexes, times=times)
    
    @classmethod
    def _sort_photons(cls, origin:PhotonDataS, statepath:Param=None, bursts:Param=None, 
                      **kwargs)->dict[str:np.ndarray[np.ndarray]]:
        """Sort photons, or retrive cached version and return dictionary"""
        # Sort PhotonDataList
        if isinstance(origin, PhotonDataList):
            # Sort PhotonDataList by sorting each sub-data and combining
            arrays = cls._sort_photons(origin.datas[0], statepath=statepath, bursts=bursts, **kwargs)
            for d in origin.datas[1:]:
                temp = cls._sort_photons(d, statepath=statepath, bursts=bursts, **kwargs)
                for k, v in temp.items():
                    arrays[k] = np.concatenate([arrays[k], v])
            return arrays
        # sort PhotonData
        kwargs = {k:v for k, v in kwargs.items() if v is not None}
        if (statepath is None) == (bursts is None):
            raise TypeError("Must specify either statepath or base, cannot specify both or neither")
        if bursts is None:
            if statepath.tp != cls:
                raise TypeError(f"Can only sort photons for param of type {cls.__name__}, got {statepath.tp.__name__}")
            pkws = tupledict(*((k, v) for k, v in statepath.params.items() if k != 'model'))
            bursts = statepath
        else:
            pkws, _ = cls.param_sort_process(kwargs, origin.detdef)
            pkws = tupledict(*((pdef.name, pkws[pdef.name]) for pdef in cls.param_defs 
                              if pdef.name in pkws and pdef.name != 'model'))
        bursts = bursts.base_param
        rkws = tupledict(*tuple((k,v) for k,v in pkws.items())+(('bursts', bursts),))
        sort_data = {name:origin._get_from_cache(cls, rkws, name) for name in 
                     chain(('indexes', 'times'), cls._sort_store)}
        if all(v is not None for v in sort_data.values()):
            return sort_data
        sort_data = cls._sort_photons_func(origin, bursts, **pkws)
        for name, array in sort_data.items():
            origin._add_to_cache(cls, pkws, name, array)
        return sort_data

    @classmethod
    def sort_photons(cls, origin:PhotonDataS, statepath:Param=None, 
                     bursts:Param=None, streams:Sequence[PhSel]=None
                     )->dict[str:np.ndarray[np.ndarray]]:
        """
        Sort photons into indexes/times arrays for processing with |H2MM|.

        Parameters
        ----------
        origin : PhotonDataS
            Data from which to sort photons.
        statepath : Param, optional
            Definition of streams, if used cannot use bursts or streams kwargs. 
            The default is None.
        bursts : Param, optional
            bursts definition, must be used with streams, and . The default is None.
        streams : Sequence[PhSel], optional
            Streams to inlcude in |H2MM| processing. The default is None.

        Returns
        -------
        dict[str:np.ndarray[np.ndarray]]
            Dictionary of sorted photons, each key contains a particular sort type.
            has the following keys:
            
            - indexes : np.ndarray[np.ndarray[np.uint8]]
              Photon indexes for |H2MM| processing.
            - times : np.ndarray[np.ndarray[np.int64]]
              Photon arrival times for |H2MM| processing

        """
        return cls._sort_photons(origin, statepath=statepath, bursts=bursts, streams=streams)

    def phsel_select(self, phsel:PhSel, col:str, fill:Any, dtype:np.dtype)->np.ndarray[np.object_]:
        r"""
        Maps the of the inputs to |H2MM| evaluation to the "unprocessed" output shape.

        Parameters
        ----------
        phsel : PhSel
            A phsel object defining the output streams to return.
        col : str
            Name of column being returned.
        fill : Any
            Value to fill any photons that are in phsel but outside of phsel_span.
        dtype : np.dtype
            Data-type of output array.

        Returns
        -------
        np.ndarray[np.object\_]
            If implemented should return object array of column maped to phsel

        """
        out = np.empty(self.size, dtype=np.object_)
        arr_type = _echo if dtype is None else partial(_astype, dtype)
        for i, (arr, ms, md) in enumerate(zip(self.iter_column(col), 
                                              self.base_table.iter_column('ph_mask', self.phsel_span),
                                              self.base_table.iter_column('ph_mask', phsel))):
            out[i] = _mask_expand(arr_type(arr), ms, md, fill)
        return out

    def _get_detpath(self):
        """Return dets submitted to H2MM, in order submitted to H2MM"""
        return self.base_table['ph_dets', self.phsel_span]
    
    @parammethod(origin_as_kw=True)
    def model_streams(cls, statepath:Param, phsel:PhSel, origin=None, strict:bool=True)->np.ndarray[np.int64]:
        """
        Determine index(es) of ``phsel`` in the :class:`hm.h2mm_model` based on
        a given param definition.


        Parameters
        ----------
        statepath : Param
            :class:`smfbursts.datamodel.tables.Param` of type :class:`smfburts.datamodel.tables.StatePath` with model.
        phsel : PhSel
            Stream selection to map to model.
        origin : PhotonData, optional
            If specified, the :class:`smfbursts.photondata.PhotonData` object
            from which ``statepath`` is assumed to have been optimized. 
            The default is None.
        strict : bool, optional
            Whether to check if the ``phsel`` contains only streams specified
            in the streams param of ``statepath``. The default is True.

        Raises
        ------
        ValueError
            ``phsel`` contains streams not used in ``statepath``.

        Returns
        -------
        used_streams : np.ndarray[np.int64]
            Array of indexes which contribute to ``phsel`` in ``statepath.model.obs``.
            ``statepath.model.obs[:,used_strams].sum(axis=1)`` will return the
            probability that a photon arises from ``phsel`` per state in the 
            ``statepath.model``.

        """
        streams, detdef = statepath.params['streams'], statepath.detdef
        phsel = phsel.render_positive(detdef, convert_all=True)
        used_streams = np.array([i for i, sphsel in enumerate(streams) if sphsel in phsel])
        if strict:
            sunion = phsel_union(*(streams[i] for i in 
                                   used_streams)).render_positive(detdef, convert_all=True)
            if phsel != sunion:
                sdif = (phsel - sunion).render_positive(detdef, convert_all=True)
                raise ValueError(f"Cannot infer {phsel} from model, model missing streams {sdif}")
        return used_streams

    @classmethod
    def param_model(cls, statepath:Param, origin:PhotonDataS=None)->hm.h2mm_model:
        """
        Return the |H2MM| model used by statpath

        Parameters
        ----------
        statepath : Param
            StatePath Param from which to retrieve |H2MM| model.
        origin : PhotonDataS, optional
            Data on which model will be used/is based. The default is None.

        Returns
        -------
        H2MM_C.h2mm_model
            Raw |H2MM| model.

        """
        return statepath.params['model']

    @parammethod(origin_as_kw=True)
    def param_streams(cls, statepath:Param, origin:PhotonDataS=None)->Sequence[PhSel]:
        """
        Return the streams used by statepath

        Parameters
        ----------
        statepath : Param
            StatePath Param from which to retrieve photon streams.
        origin : PhotonDataS, optional
            Data on which model will be used/is based. The default is None.

        Returns
        -------
        Sequence[PhSel]
            Sequence of PhSel defining streams of each |H2MM| index.

        """
        return statepath.params['streams']    

    @classmethod
    def optimize(cls, origin:PhotonDataS, bursts:Param, model:hm.h2mm_model,
                 streams:Sequence[PhSel]=None, gate:GateGroup=None, **kwargs)->Param:
        """
        Optimize a model against data defined by bursts in origin, using indexes
        defined by streams

        Parameters
        ----------
        origin : PhotonDataS
            Source of burst data.
        bursts : Param
            Time (and thus photon arrays) defining data to optimize.
        model : hm.h2mm_model
            Initial model to start optimization.
        streams : Sequence[PhSel], optional
            Sequence of PhSel, defines indexes given to |H2MM|. The default is None.
        gate : GateGroup, optional
            Gate to apply to bursts. The default is None
        **kwargs : Any
            Additional kwargs handed to model.optimize.

        Returns
        -------
        Param
            StatePath based Param of optimized model.

        """
        # get param w/gate to run optimization
        bursts = bursts.base_param
        bursts = bursts if gate is None else bursts.regate(gate)
        # process streams
        if streams is None:
            streams = tuple(origin.detdef.stream_ids_to_PhSel(i, convert_all=True) 
                            for i in range(origin.detdef.size))
        pkws = {'streams':streams}
        pkws.update({pdef.name:kwargs.pop(pdef.name, None) for pdef in cls.param_defs 
                     if pdef.name not in ('model', 'streams')})
        pkws = {k:v for k, v in pkws.items() if v is not None}
        arrays = cls._sort_photons(origin, bursts=bursts, **pkws)
        index, times = arrays['indexes'], arrays['times']
        index, times = _empty_remove(index), _empty_remove(times)
        # run optimization
        mopt = model.optimize(index, times, inplace=False, **kwargs)
        pkws.update(model=mopt)
        flag = {'h2mm':{'loglik':mopt.loglik, 'nphot':mopt.nphot, 'k':mopt.k, 
                        'bic':mopt.bic, 'conv_code':mopt.conv_code}}
        return Param(cls, params=pkws, parents={'bursts':bursts}, flags={('dataID'+origin.dataID):flag})

    @classmethod
    def _model_iter(cls, ndet:int, index:DArray, times:TArray, start:int, stop:int, 
                    models=None, bounds=None, trans_scale=1e-5, 
                    prior_dist='equal', trans_dist='equal', obs_dist='equal',
                    **kwargs)->Iterator[hm.h2mm_model]:
        """Iterator from start to stop of optimized models"""
        modelget = _ModelGetter(models)
        for i in range(start, stop+1):
            model = modelget.get(i)
            if model is None:
                model = hm.factory_h2mm_model(i, ndet, bounds=bounds, trans_scale=trans_scale, 
                                              prior_dist=prior_dist, trans_dist=trans_dist, 
                                              obs_dist=obs_dist)
            yield model.optimize(index, times, bounds=bounds, **kwargs)

    @classmethod
    def optimize_models(cls, origin:PhotonDataS, bursts:Param, streams:Sequence[PhSel]=None,
                 min_states:int=1, max_states:int=8, to_state:int=4,
                 conv_crit:CCrit='pathBIC', thresh:None|float=None, 
                 gate:GateGroup=None, **kwargs)->list[Param]:
        r"""
        Given data and burst definition, optimize successive state-models and
        return the cooresponding state-path ``Param`` s in a list.

        Parameters
        ----------
        origin : PhotonDataS
            Source of photon data to optimize.
        bursts : Param
            Time ranges to optimize.
        streams : Sequence[PhSel], optional
            Sequence of PhSel, defines indexes given to |H2MM|. 
            If None, use each detector ID as separate stream.
            The default is None.
        min_states : int, optional
            Number of states in first model optimized. The default is 1.
        max_states : int, optional
            Maximum number of states, regardless of ``conv_crit`` in models 
            to optimize, sets an upper limit of the number of states to optimize. 
            The default is 8.
        to_state : int, optional
            Minimum number of states in models guaranteed to optimize, regardless
            of ``conv_crit``, it is guaranteed that the returned sequence will
            contain StatePaths with models with from min_states to to_state 
            (inclusize) numbers of states. The default is 4.
        conv_crit : Literal['BIC','BICp','BICph','ICL','ICLph','pathBIC','pathBICph'] | Callable, optional
            Criterion to use to determine of best model has been found,
            the convergence criterion. This can either be a string defining the
            function or a callable. Options are\:
                
                - 'BIC' Choose based on BIC, uses :func:`conv_BIC`
                - 'BICp' Choose based on modified BIC, uses :func:`conv_BICp`
                - 'BICph' Choose based on BIC per photon, uses :func:`conv_BICph`
                - 'ICL' Choose based on ICL, uses :func:`conv_ICL`
                - 'ICLph' Choose based on ICL per photon, uses :func:`conv_ICLph`
                - 'pathBIC' Choose based on the BIC of most likely state path, 
                   uses :func:`conv_pathBIC`
                - 'pathBICph' Choose based on BIC of most likely state path,
                  uses :func:`conv_pathBICph`
            
            If specified as a callable, should have a signature like that of 
            :func:`conv_BIC` ie
            ``conv_crit(models:Sequence[hm.h2mm_model], origin:PhotonDataS, params:Sequence[Param], thresh:None|float)->int``
            The default is 'pathBIC'.
        thresh : float, optional
            If specified, the threshold to consider a model converged, ie if the
            difference in the statistical disciminator between a model and the minimum
            is less than thresh, then that model is considered ideal. The default is None.
        gate : GateGroup, optional
            Gate to apply to bursts. The default is None.
        **kwargs : Any
            Additional kwargs handed to ``model.optimize()`` .

        Returns
        -------
        list[Param]
            List of StatePath based Params of optimized models.

        """
        bursts = bursts.base_param
        bursts = bursts if gate is None else bursts.regate(gate)
        if streams is None:
            streams = tuple(origin.detdef.stream_ids_to_PhSel(i, convert_all=True) 
                            for i in range(origin.detdef.size))
        pkws = {pdef.name:kwargs.pop(pdef.name, None) for pdef in 
                cls.param_defs if pdef.name not in ('model', 'streams')}
        pkws['streams'] = streams
        arrays = cls._sort_photons(origin, bursts=bursts, **pkws)
        index, times = arrays['indexes'], arrays['times']
        index_, times_ = _empty_remove(index), _empty_remove(times)
        models:list[hm.h2mm_model] = list()
        params:list[Param] = list()
        conv_crit = cls._conv_funcs[conv_crit] if isinstance(conv_crit, str) else conv_crit
        for model in cls._model_iter(cls.param_ndet(pkws), index_, times_, min_states, max_states, **kwargs):
            if np.any(np.isnan(model.prior)) or np.any(np.isnan(model.trans)) or np.any(np.isnan(model.obs)):
                break
            models.append(model)
            pkws.update(model=model)
            flag = {'h2mm':{'loglik':model.loglik, 'nphot':model.nphot, 'k':model.k, 
                            'bic':model.bic, 'conv_code':model.conv_code}}
            params.append(Param(cls, params=pkws, parents={'bursts':bursts}, 
                                flags={('dataID'+origin.dataID):flag}))
            if model.nstate >= to_state and conv_crit(models, origin, params, thresh) != -1:
                break
        return params


def _append_beg(val, arr):
    """Append val to beginning of array, faster than concatentate"""
    out = np.empty(arr.size+1, dtype=arr.dtype)
    out[0] = val
    out[1:] = arr
    return out


def _append_end(arr, val):
    """Append val to end of array, faster than concatenate"""
    out = np.empty(arr.size+1, dtype=arr.dtype)
    out[:-1] = arr
    out[-1] = val
    return out


# @fnumba.jit(fnumba.int8[:](fnumba.int64[:]))
def dwell_distsince(burst_num:np.ndarray[np.int64])->np.ndarray[np.int8]:
    """
    Find number of dwells since the last burst based on burst number of dwells.

    Parameters
    ----------
    burst_num : np.ndarray[np.int64]
        Burst number of each dwell.

    Returns
    -------
    dwell_f : np.ndarray[np.int8]
        Number of dwells since last new burst.

    """
    dwell_f = np.empty(burst_num.size, dtype=np.int8)
    n, prev = 0, -1
    for i, p in enumerate(burst_num):
        if p == prev:
            n += 1
        else:
            n = 0
        dwell_f[i] = n
        prev = p
    return dwell_f


# @fnumba.jit(fnumba.int8[:](fnumba.int64[:]))
def dwell_distuntil(burst_num):
    """
    Find number of dwells until the next burst based on burst number of dwells.

    Parameters
    ----------
    burst_num : np.ndarray[np.int64]
        Burst number of each dwell.

    Returns
    -------
    dwell_r : np.ndarray[np.int8]
        Number of dwells until next new burst.

    """
    dwell_r = np.empty(burst_num.size, dtype=np.int8)
    n, prev = 0, -1
    for i in range(burst_num.size-1,-1,-1):
        if burst_num[i] == prev:
            n += 1
        else:
            n = 0
        dwell_r[i] = n
        prev = burst_num[i]
    return dwell_r


def dwell_distmin(burst_num:np.ndarray[np.int64])->np.ndarray[np.int8]:
    """
    Minimum number of dwells to reach new burst (from or until)

    Parameters
    ----------
    burst_num : np.ndarray[np.int64]
        Burst number of each dwell.

    Returns
    -------
    dwell_min : np.ndarray[np.int8]
        Minimum number of dwells (from or until) to reach new burst.

    """
    return np.min(np.vstack([dwell_distsince(burst_num),dwell_distuntil(burst_num)]), axis=0).astype(np.int8)


def dwell_distmax(burst_num:np.ndarray[np.int8])->np.ndarray[np.int8]:
    """
    Maximum number of dwells to reach transition (from or until)

    Parameters
    ----------
    burst_num : np.ndarray[np.int64]
        Burst number of each dwell.

    Returns
    -------
    dwell_max : np.ndarray[np.int8]
        Maximum number of dwells (from or until) to reach a new burst.

    """
    return np.max(np.vstack([dwell_distsince(burst_num),dwell_distuntil(burst_num)]), axis=0).astype(np.int8)


def dwell_distminneg(burst_num:np.ndarray[np.int64])->np.ndarray[np.int8]:
    """
    Minimum number of dwells to a different burst, with sign indicating direction.
    If nearest different burst is before dwell, value is positive, if nearest transition
    is after dwell, then value is negative, **and shifted by -1** ie if a dwell is
    1 transition away from the next burst, it will have a value of :math:`-2 = -1 + -1`,
    This means that the ending dwell of a burst has a value of -1.

    Parameters
    ----------
    burst_num : np.ndarray[np.int64]
        Burst number of each dwell.

    Returns
    -------
    dwell_pos : np.ndarray[np.int8]
        Signed minimum number of dwells to new burst.

    """
    dwell_pos, dwellr = dwell_distsince(burst_num), dwell_distuntil(burst_num)
    mask = dwellr < dwell_pos
    dwellr *= -1
    dwellr -= 1
    dwell_pos[mask] = dwellr[mask]
    return dwell_pos


def dwell_distminpos(burst_num:np.ndarray[np.int64])->np.ndarray[np.int8]:
    """
    Determine dwell "position" in burst, negative values indicate beginning or ending
    dwells, while positive values are minimiumn number of dwells to reach new burst
    (ie same values as would be returned by :func:`dwell_distfrmin`).
    -1 indicates a beginning dwell, -2 an ending dwell, and -3 a dwell that is both
    beginning and ending (whole burst dwell).

    Parameters
    ----------
    burst_num : np.ndarray[np.int64]
        Burst number of each dwell.

    Returns
    -------
    burst_min : np.ndarray[np.int8]
        Burst position of each .

    """
    dwell_pos = dwell_distmin(burst_num)
    mask = np.diff(burst_num) != 0
    dwell_pos[0] -= 1
    dwell_pos[-1] -= 2
    dwell_pos[1:][mask] -= 1
    dwell_pos[:-1][mask] -= 2
    return dwell_pos


_dwell_posfunction = {'since':dwell_distsince, 'until':dwell_distuntil, 
                      'min':dwell_distmin, 'max':dwell_distmax,
                      'nmin':dwell_distminneg, 'pos':dwell_distminpos}

#: TypeValidator for string of dwell_pos column first key
TV_str_dwelldist = TV_str(isin=tuple(_dwell_posfunction.keys()))


def _title_dwell_startstop_append(name:str, start:str, stop:str)->str:
    if start == 'start' and stop == 'stop':
        name += r'\: full'
    elif start != 'rstart' or stop != 'rstop':
        name += rf'\: [{start},{stop}]'
    return name

#: TypeValidator for start time type in Dwells
TV_str_dwellstart = TV_str(isin=('istart', 'istarttime', 'rstart'))
#: TypeValidator for stop time type in Dwells
TV_str_dwellstop = TV_str(isin=('istop', 'istoptime', 'rstop'))


class Dwells(BasePhotonTable):
    """
    Defines ranges of time inside of burst that have consecutive photons in the
    same most-likely state, as determined by the *Viterbi* algorithm.
    
    Params
    ------
    No params exist, Dwells are fully defined by parent statepath
    
    Parents
    -------
        statepath : StatePathBase
            Any :class:`Param` based on a subclass of :class:`StatePathBase`
            (defines bursts and H2MM model) for which to segment state-paths
            into dwells.
    
    Columns
    -------
        Dwells is a :class:`BasePhotonTable` and thus all columns for 
        :class:`BasePhotonTable` are also present in Dwells.
        
        The following additional columns are also defined:
        
        rstart : int, ()
            "Propper" start of a dwell. If dwell is in middle of burst, take
            midpoint between photons of the transition, if at the beginning
            of a burst, take the first photon of that burst as the start time.
        rstop : int, ()
            "Propper" stop of a dwell. If dwell is in middle of burst, take
            midpoint between photons of the transition, if at the end of a burst,
            take the last photon of that burst as the stop time.
        state : np.int8, (i:int, )
            State of dwell shifted by i, if that dwell is in another burst, takes
            value of -1. 
            If i is 0, then state is state of dwell.
            If i is negative, then the state of the dwell abs(i) dwells before
            current dwell. 
            If i is possitive, state of i dwells after the current
            dwell.
        iburst : int, ()
            The index of the burst to which the dwell belongs based on statepath
            (including gates)
        dwell_pos : np.int8, ({'since', 'until', 'min', 'max', 'nmin', 'pos'}, )
            The position of the dwell within a burst. Key indicates direction/function
            used to evaluate.
            
                - 'since' number of dwells to begginning of burst
                - 'until' number of dwells to end of burst
                - 'min' min[since, until]
                - 'max' max[since, unitl]
                - 'nmin' signed min[since, unil] (if maximum is since, value is negative and shifted by -1)
                - 'pos' "position" of dwell in burst. Beginning/ending/whole burst
                  dwells have values of -1, -2, and -3 respectively, otherwise
                  give result of min
                  
        nph_h2mm : int, (phsel:PhSel)
            Number of photons based on non-reordered statepath
        ratio_h2mm : float, (phsel_num:Phsel, phsel_dem:PhSel)
            Ratio of number of photons in phsel_num to phsel_dem 
            based on non-reordered statepath
        anisotropy_h2mm : float, (phsel_p:Phsel, phsel_s:PhSel)
            Anisotropy of number of photons in phsel_p vs phsel_s 
            based on non-reordered statepath
        
    """
    _colstarttype = TV_str_dwellstart
    _colstoptype = TV_str_dwellstop
    _colstartdefault = 'rstart'
    _colstopdefault = 'rstop'
    #: :meta private:
    param_defs = tuple()
    #: :meta private:
    parent_defs = (
        ParentDef('statepath', StatePathBase),
                   )
    _parent_ph_subrange = 'statepath'
    #: :meta private:
    column_defs = make_base_column_defs(TV_str_dwellstart, TV_str_dwellstop) + (
        ColumnDef('rstart', tuple(), 0, 'never', dtype=np.dtype('<i8'), 
                  get_func='_get_rstart', title='rstart', unit='clk_p'),
        ColumnDef('rstop', tuple(), 0, 'never', dtype=np.dtype('<i8'), 
                  get_func='_get_rstop', title='rstop', unit='clk_p'),
        ColumnDef('state', (int, ), 0, 'some', dtype=np.dtype('<i1'), 
                  reg_func='_regularizecolumn_state', get_func='_get_state'),
        ColumnDef('iburst', tuple(), 0, 'all', dtype=np.dtype('<i8'), title='burst index'),
        ColumnDef('dwell_pos', (str,), 0, 'user', dtype=np.dtype('<i1'), reg_func='_regularizecolumn_dwell_pos',
                  get_func='_get_dwell_pos', get_derived=True, check_func='_check_dwell_pos'), # for defining where in burst dwell is
        ColumnDef('nph_h2mm', (PhSel,), 0, 'user', get_derived=True, dtype=np.dtype('<i8'),
                  iter_func='_iter_nph_h2mm', unit='cnts', check_func='_check_nph_h2mm'),
        ColumnDef('ratio_h2mm', (PhSel, PhSel), 0, 'user', get_derived=True,
                  get_func='_get_ratio_h2mm', check_func='_check_ratio_h2mm'),
        ColumnDef('anisotropy_h2mm', (PhSel, PhSel), 0, 'user', get_derived=True,
                  get_func='_get_anisotropy_h2mm', check_func='_check_anisotropy_h2mm'),
        ColumnDef('nanohist_thresh', (PhSel, float, str, bool), 0, 'never',
                  iter_func='_iter_nanohist_thresh', reg_func='_regularizecolumn_nanohist_thresh',
                  get_derived=False, dtype=np.dtype('<i8'), ndim=2),
        )

    @paramproperty
    def detdef(cls, param:Param):
        return param.parents['statepath'].detdef

    def __init_columns__(self):
        patht = self.parents['statepath']
        baset = patht.base_table
        starts = np.empty(baset.size, dtype=np.object_)
        stops = np.empty(baset.size, dtype=np.object_)
        iburst = np.empty(baset.size, dtype=np.object_)
        states = np.empty(baset.size, dtype=np.object_)
        for i, (start, stop, times, state) in enumerate(zip(baset.iter_column('start'), 
                                                            baset.iter_column('stop'), 
                                                            patht.iter_column('timepath'), 
                                                            patht.iter_column('statepath'))):
            loc = np.diff(state) != 0
            mids = (times[:-1][loc]+times[1:][loc])//2
            starts[i] = _append_beg(start, mids)
            stops[i] = _append_end(mids, stop)
            iburst[i] = np.ones(starts[i].shape, dtype=np.int64)*i
            states[i] =  _append_beg(state[0], state[1:][loc]).astype(np.int8)
        starts, stops = np.concatenate(starts), np.concatenate(stops)
        iburst, states = np.concatenate(iburst), np.concatenate(states)
        istarts, istops = smc.index_ranges(self.origin.times, starts, stops)
        self._add_column('istart', tuple(), istarts)
        self._add_column('istop', tuple(), istops)
        self._add_column('start', tuple(), starts)
        self._add_column('stop', tuple(), stops)
        self._add_column('iburst', tuple(), iburst)
        self._add_column('state', (0, ), states)

    def _get_rstart(self)->np.ndarray[np.int64]:
        """
        Get the start time of dwell, if dwell is the first in a burst, 
        return first photon, otherwise return midpoint between last photon of
        previous dwell, and first photon of curent dwell.
        """
        out, istops = self['start'].copy(), self['istarttime']
        out[0] = istops[0]
        mask = np.diff(self['iburst']) != 0
        out[1:][mask] = istops[1:][mask]
        return out

    def _get_rstop(self)->np.ndarray[np.int64]:
        """
        Get the stop time of dwell, if dwell is the last in a burst, 
        return last photon, otherwise return midpoint between last photon of
        current dwell, and first photon of the next dwell.
        """
        out, istops = self['stop'].copy(), self['istoptime']
        out[-1] = istops[-1]
        mask = np.diff(self['iburst']) != 0
        out[:-1][mask] = istops[:-1][mask]
        return out

    @classmethod
    def _regularizecolumn_state(cls, source_param:Param, *args):
        return args if args else (0, )

    @classmethod
    def _title_startstop_append(cls, name:str, start:str, stop:str)->str:
        return _title_dwell_startstop_append(name, start, stop)

    def _state_shift(self, i:int, fill:Any, array:np.ndarray)->np.ndarray:
        if i == 0:
            return array
        slc = slice(-i,None) if i < 0 else slice(None, -i)
        islc = slice(None,i) if i < 0 else slice(i, None)
        iburst = self['iburst',]
        out = np.empty(array.shape, dtype=array.dtype)
        out[slc] = array[islc]
        mask = np.ones(array.shape, dtype=np.bool_)
        mask[slc] = iburst[slc] != iburst[islc]
        if out.dtype != np.object_:
            out[mask] = fill
        else:
            for j in range(out.size):
                if mask[j]:
                    out[j] = fill
        return out

    @classmethod
    def _regularizecolumn_shift(cls, *args)->tuple[str,]:
        if not args:
            return 'fwd'

    def _get_state(self, i:int)->np.ndarray[np.ndarray[np.int8]]:
        out = self._bcache['state', 0]
        if i == 0:
            return out
        return self._state_shift(i, -1, out)

    def _get_dwell_pos(self, direction:str)->np.ndarray[np.int8]:
        return _dwell_posfunction[direction](self['iburst'])

    @classmethod
    def _check_dwell_pos(cls, col:Column)->None:
        if col.keytup[0] not in _dwell_posfunction:
            raise ValueError(f"dwell_pos, '{col.keytup[0]}' invalid, "+
                             "second string must be in '"+
                             "', '".join(_dwell_posfunction.keys()))

    @classmethod
    def _regularizecolumn_dwell_pos(cls, source_param:Param, *args)->tuple[str,]:
        return args if args else ('pos', )

    def _iter_nph_h2mm(self, phsel:PhSel)->Iterator[int]:
        stream_ids = self.param.detdef.get_stream_ids(phsel)
        ptable = self.parents['statepath']
        startstopiter = zip(self.iter_column('start'), self.iter_column('stop'))
        start, stop = next(startstopiter)
        for times, dets in zip(ptable.iter_column('timepath'), ptable.iter_column('detpath')):
            while True:
                if start > times[-1]:
                    break
                yield np.isin(dets[(start < times) & (times < stop)], stream_ids).sum()
                try:
                    start, stop = next(startstopiter)
                except StopIteration:
                    return

    @classmethod
    def _check_nph_h2mm(self, col:Column):
        if col.keytup[0] not in phsel_union(col.param.parents['statepath'].params['streams']):
            raise ValueError("nph_h2mm phsel must be within streams of statepath parent")

    def _get_ratio_h2mm(self, phsel_num:PhSel, phsel_dem:PhSel)->np.ndarray[np.float64]:
        return self['nph_h2mm', phsel_num] / self['nph_h2mm', phsel_dem]

    @classmethod
    def _check_ratio_h2mm(self, col:Column):
        phsel_span = phsel_union(col.param.parents['statepath'].params['streams'])
        if any(phsel not in phsel_span for phsel in col.keytup):
            raise ValueError("ratio_h2mm phsel must be within streams of statepath parent")

    def _get_anisotropy_h2mm(self, phsel_p:PhSel, phsel_s:PhSel)->np.ndarray[np.float64]:
        p, s = self['nph_h2mm', phsel_p], self['nph_h2mm', phsel_s]
        return (p-s)/(p+2*s)

    def _check_anisotropy_h2mm(self, col:Column):
        phsel_span = phsel_union(col.param.parents['statepath'].params['streams'])
        if any(phsel not in phsel_span for phsel in col.keytup):
            raise ValueError("anisotropy_h2mm phsel must be within streams of statepath parent")
    
    def _iter_nanohist_thresh(self, phsel:PhSel, thresh:float, discr:Literal['ph_gamma', 'ph_scale', 'ph_pathll'], full:bool):
        if not phsel.positive:
            phsel = phsel.render_positive(self.origin.detdef, convert_all=False)
        if full:
            mn = 0
            ln = np.max(self.origin.setup.tcspc_num_bins)
        else:
            elements = phsel.ex.elements if phsel.ex.kind else (i for i in range(self.detdef.ex) if i not in phsel.ex.elements)
            ex_ranges = np.concatenate([self.origin.setup.ex_ranges[i] for i in elements])
            mn, mx = np.min(ex_ranges), np.max(ex_ranges)
            ln = mx - mn
        ptable = self.parents['statepath']
        btable = ptable.base_table
        startstopiter = zip(self.iter_column('start'), self.iter_column('stop'))
        start, stop = next(startstopiter)
        if discr == 'ph_gamma':
            for times, nanos, post, state in zip(btable.iter_column('ph_times', phsel), 
                                                 btable.iter_column('ph_nanos', phsel), 
                                                 ptable.iter_column(discr, phsel),
                                                 ptable.iter_column('ph_state', phsel)):
                if discr == 'ph_gamma':
                    post = post[:,state]
                mask = thresh <= post[range(post.shape[0], state)]
                while True:
                    if start > times[-1]:
                        break
                    yield np.bincount(nanos[(start < times) & (times < stop) & mask]-mn, minlength=ln)
                    try:
                        start, stop = next(startstopiter)
                    except StopIteration:
                        return
        else:
            for times, nanos, post in zip(btable.iter_column('ph_times', phsel), 
                                          btable.iter_column('ph_nanos', phsel), 
                                          ptable.iter_column(discr, phsel)):
                mask = thresh <= post
                while True:
                    if start > times[-1]:
                        break
                    yield np.bincount(nanos[(start < times) & (times < stop) & mask]-mn, minlength=ln)
                    try:
                        start, stop = next(startstopiter)
                    except StopIteration:
                        return

    @classmethod
    def _regularizecolumn_nanohist_thresh(cls, source_param:Param, *args)->tuple[PhSel,float,Literal['ph_gamma', 'ph_scale', 'ph_pathll'], bool]:
        """Column regularization function for nanohist_thresh column"""
        if not args:
            raise TypeError("must specify at least phsel of nanohist_state")
        if len(args) > 4:
            raise TypeError("too many keys for nanohist_state, maximumn 4, PhSel and full, (full optional)")
        if not isinstance(args[0], PhSel):
            raise TypeError("must specify PhSel as first key in nanohist_state column")
        out = [args[0],]
        i = 1
        for tp, cast, default in ((Real, float, 0.0), (str, str, 'ph_gamma'), (bool, bool, False)):
            if i < len(args) and isinstance(args[i], tp):
                out.append(cast(args[i]))
                i += 1
            else:
                out.append(default)
        if i != len(args):
            raise ValueError("Unrecognized keys: {args")
        if out[2] not in ('ph_scale', 'ph_pathll', 'ph_gamma'):
            raise TypeError(f"dicr must be either 'ph_gamma', 'ph_scale' or 'ph_pathll', not {out[2]}")
        if out[2] == ('ph_scale', 'ph_pathll'):
            if out[1] > 0.0:
                raise ValueError("'{out[2]}' type thresholds must be less than 0")
        elif out[1] < 0.0 or out[1] > 1.0:
            raise ValueError(f"'{out[2]}' type thresholds must be greater than 0")
        return tuple(out)

###############################################################################
######## Functions/classes for processing nanotime divisor based data  ########
###############################################################################
def sort_index_time_div(id_map:np.ndarray[np.int8], dets:np.ndarray[np.uint8], 
                         times:np.ndarray[np.int64], nanos:np.ndarray[np.uint16],
                         divs:np.ndarray[np.uint16])->tuple[np.ndarray[np.uint8],np.ndarray[np.int64],np.ndarray[np.bool_]]:
    r"""
    Single burst sort function.
    Sort photons into function returns input for ``hm.h2mm_model.optimize``
    which splits detectors by divisions from nanotime. Usually called internally
    by :func:`sort_indexes_times_divs`

    Parameters
    ----------
    id_map : np.ndarray[np.int8]
        Detector index map, maps dets to group.
    dets : np.ndarray[np.uint8]
        Original detector indexes.
    times : np.ndarray[np.int64]
        Times of photons.
    nanos : np.ndarray[np.uint16]
        Nanotimes.
    divs : np.ndarray[np.uint16]
        Per stream in id_map, divisions within nanotimes.

    Returns
    -------
    ndets : np.ndarray[np.uint8]
        Indexes to submit to ``hm.h2mm_model.optimize()``.
    times : np.ndarray[np.int64]
        Times to submit to ``hm.h2mm_model.optimize()``.
    mask : np.ndarray[np.bool\_]

    """
    dets = id_map[dets]
    ndets = np.empty(dets.shape, dtype=np.uint8)
    shift = 0
    for i, div in enumerate(divs):
        mask = dets == i
        ndets[mask] = shift
        for thresh in div:
            ndets += ((nanos >= thresh) & mask)
            shift += 1
        shift += 1
    return ndets, times, dets >= 0


@cite('HarrisBioPhysRep2022', purpose='Divisor (lifetime) based H2MM optimization')
def sort_indexes_times_divs(table:BasePhotonTable, streams:Sequence[PhSel], 
                       divs:np.ndarray[np.object_]
                       )->tuple[np.ndarray[np.ndarray[np.uint8]],np.ndarray[np.ndarray[np.int64]]]:
    r"""
    Retreive arrays of bursts information for optimization in |H2MM| from a
    ``BasePhotonTable`` using the divisor approach
    (`Harris 2022 <https://doi.org/10.1016/j.bpr.2022.100071>`_ )

    Parameters
    ----------
    table : BasePhotonTable
        Table to retrieve |H2MM| inputs from.
    streams : Sequence[PhSel]
        Streams to include as distinct indexes (further subdivided by divs) in output.
    divs : np.ndarray[np.object\_]
        Per photon stream, nanotime thresholds to use as divisions creating new index.

    Returns
    -------
    out_dets : np.ndarray[np.ndarray[np.uint8]]
        Indexes to submit to ``hm.h2mm_model.optimize()``.
    out_times : np.ndarray[np.ndarray[np.int64]]
        Times to submit to ``hm.h2mm_model.optimize()``.

    """
    phselu = phsel_union(*streams)
    id_map = reindex_phsel(table.origin.detdef, streams)
    out_dets = np.empty(table.size, dtype=np.object_)
    out_times = np.empty(table.size, dtype=np.object_)
    if not table.origin.save_memory and hasattr(table.origin, 'times'):
        times, nanos = table.origin.times, table.origin.nanos
        ndets, times, mask = sort_index_time_div(id_map, table.origin.dets, times, nanos, divs)
        for i, (istart, istop) in enumerate(zip(table.iter_column('istart'), 
                                                table.iter_column('istop'))):
            out_dets[i] = ndets[istart:istop][mask[istart:istop]]
            out_times[i] = times[istart:istop][mask[istart:istop]]
    else:
        biter = zip(table.iter_column('ph_dets', phselu), 
                    table.iter_column('ph_times', phselu), 
                    table.iter_column('ph_nanos', phselu), repeat(divs))
        for i, inp in enumerate(biter):
            ndets, ntimes, mask = sort_index_time_div(id_map, *inp)
            out_dets[i], out_times[i] = ndets[mask], ntimes[mask]
    return out_dets, out_times


class ntdivStatePath(StatePath):
    """
    Table of statepath where photons are processed using the |divisorapproach|.
    In addition to the standard :class:`StatePath` parameters, it also takes
    the ``divs`` parameter, which defines the position of divisors in each stream.
    
    With the inclusion of divisors, it is possible with ``ntdivStatePath`` to 
    compute model nanomean values with :meth:`StatePath.model_value`. 
    :meth:`StatePath.model_value` and :meth:`StatePath.model_values` methods.
    
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
        
    Parents
    -------
        bursts : Param[BasePhotonTable]
            Usually a :class:smfbursts.datamodel.tables.Bursts` 
            :class:`smfbursts.datamodel.tables.Param`, defining the time 
            ranges of each "burst" in |H2MM| processing.
    
    Columns
    -------
    Note that these are identical to :class:`StatePath`
    
        indexpath : np.ndarray[np.uint8] ()
            Actual indexes used in |H2MM| processing. Each row is uint8 array.
        detpath : np.ndarray[np.uint8] ()
            Detector indexes of photons used in |H2MM| processing, not reassigned by streams.
            Each row is uint8 array.
        timepath : np.ndarray[np.int64] ()
            Actual times used in |H2MM| processing. Each row is int64 array.
        statepath : np.ndarray[np.uint8] ()
            Most likely states of each photon as processed in *Viterbi* algorithm.
            This is the direct output of *Viterbi*, no sub-selecting photons etc.
            Each row is 1d uint8 array
        scalepath : np.ndarray[np.float64], ()
            Posterior per photon of *Viterbi* processing. This is direct output of *Viterbi*.
            Each row is 1D float64 array.
        pathllpath : np.ndarray[np.float64], ()
            Log-likelihood of most likely state of each photon. This is direct output of *Viterbi*.
            Each row is 1D float64 array.
        gammapath : np.ndarray[np.float64], ()
            Gamma array, giving likelihood per-photon per-state.
            Each row is 2d float64 array, indexed [photon, state].
        ph_index : np.ndarray[np.int8], (phsel:PhSel, )
            Indexes submitted to |H2MM| processing, mapped/masked by phsel.
        ph_h2mmtime : np.ndarray[np.int64], (phsel:PhSel, )
            Times of photons used in |H2MM| processing mapped/masked by phsel. 
            Photons outside of model-streams receive value of -1. 
            Rows are 1d int64 arrays.
        ph_state : np.ndarray[np.int8], (phsel:PhSel, )
            *Viterbi* state of each photon, mapped/masked by phsel. 
            Photons outside of model-streams receive value of -1. 
            Rows are 1d int8 arrays.
        ph_scale : np.ndarray[np.float64], (phsel:PhSel, )
            Posterior likelihood of each phton, mapped/masked by phsel.
            Photons outside of model-streams receive value of nan.
            Rows are 1d float64 arrays.
        ph_ll : np.ndarray[np.float64], (phsel:PhSel, )
            log-likihood of state-assignment of photon for *Viterbi* path, mapped/masked by phsel.
            Photons outside of model-streams receive value of nan.
            Rows are 1d float64 arrays
        ph_gamma : np.ndarray[np.float64], (phsel:PhSel, )
            Gamma array, giving likelihood per-photon per-state mapped/masked by phsel.
            Each row is 2d float64 array, indexed [photon, state].
        bstates: int, ()
            Bitcode indicating which states present in burst. If given bit position
            is present, then that state is present in burst, ie if states 0 and 2 are
            in a given burst, the the value is 0b00000101 = 5. Rows are int.
        eff_state : int, (phsel:PhSel, )
            "Effective" state of each photon in burst maped/masked by phsel. 
            If a photon is in a stream not present in model streams, then infer state 
            by nearest photon that is in streams.
            This is essentially ph_state with -1s replaced with an infered state.
            Rows are 1d int8 arrays.
    
    """
    #: meta private:
    param_defs = StatePath.param_defs + (
        ParamDef('divs', TV_tuple(typedefs=TV_ndarray(dtype=np.dtype('<u2'), 
                                                      superdtype=np.integer, mn=0,))),
        )
    _model_column_funcs = ImDict({k:v for k, v in 
                                  chain(StatePath._model_column_funcs.items(), 
                                        {'nanomean':[(BasePhotonTable, '_get_model_lifetime'), ]}.items())})

    @classmethod
    def validate_param(cls, param:Param):
        """Validate valid parameter :meta private:"""
        streams, divs = param.params['streams'], param.params['divs']
        detdef = param.detdef
        nstream = _validate_model_streams(streams, detdef)
        if nstream != (ndiv := len(divs)):
            raise ValueError(f"Inconsistent number of streams and divs, {nstream} vs {ndiv}")
        if any(np.any(np.diff(div) < 1) for div in divs):
            raise ValueError("divs must be monotonically increasing in a given stream")

    @classmethod
    def param_sort_process(cls, params:dict[str,Any], detdef:DetDef):
        """
        In a params dictonary, sort the streams so that streams are in ascending
        order based on DetDef
        
        Parameters
        ----------
        params : dict[str:Any]
            params dictionary to be used in ntdivStatePath based :class:`Param`.
        detdef : DetDef
            :class:`DetDef` of expected :class:`Param`.

        Returns
        -------
        params : dict[str,Any]
            Sorted params dictoinary.
        sort : np.ndarray[np.int64]
            Re-sorting array, value is original index, position is index for destination.
            Therefore new = old[sort]

        """
        if params.get('streams',None) is None:
            params['streams'] = tuple(detdef.stream_ids_to_PhSel(i) for i in range(detdef.size))
            sort = np.arange(detdef.size)
        else:
            params['streams'], sort = sort_phsels(params['streams'], detdef=detdef, return_index=True)
        if 'divs' not in params:
            raise TypeError("Must assign divs")
        if len(params['divs']) != len(params['streams']):
            raise ValueError("mismatched number streams and div")
        params['divs'] = tuple(np.asarray(div, dtype=np.uint16) for div in params['divs'])
        if any(div.ndim > 1 for div in params['divs']):
            raise ValueError("Divs must be 1d")
        params['divs'] = tuple(div.reshape(-1) for div in params['divs'])
        if any(np.any(np.diff(div) < 1) for div in params['divs']):
            raise ValueError("div arrays must be monotonically increasing")
        return params, sort

    @classmethod
    def param_ndet(cls, param:dict)->int:
        """Get number of parameters from Param of type ntdivStatePath"""
        return sum(div.size + 1 for div in param['divs'])

    @classmethod
    def param_idx_to_det_map(cls, params:dict[str,Any], detdef:DetDef)->np.ndarray[np.uint8]:
        """
        Returns 1D numpy array that should map 
        h2mm index to detector index- ie ``idxmap[index] = detector``
        where detector should match the ph_dets array
        
        Parameters
        ----------
        params : dict[str:Any]
            param dictionary definition for class, may omit keys not requied to
            compute the idx_to_det_map.
        detdef : DetDef
            :class:`DetDef` of data for which param is expected to be based.
            Needed to determing dets
        
        Returns
        -------
        np.ndaray[np.uint8]
            mapping of H2MM idx to det based on detdef.
        """
        return np.concatenate([np.repeat(detdef.get_stream_ids(stream)[0], div.size+1) 
                               for stream, div in zip(params['streams'], params['divs'])])

    @classmethod
    def _sort_photons_func(cls, origin:PhotonData, bursts:Param, streams:Sequence[PhSel], 
                           divs:Sequence[np.ndarray[np.uint16]])->dict[str:np.ndarray[np.ndarray]]:
        indexes, times = sort_indexes_times_divs(origin.get_table(bursts), streams, divs)
        return dict(indexes=indexes, times=times)

    @classmethod
    def sort_photons(cls, origin:PhotonDataS, statepath:Param=None, bursts:Param=None, 
                     streams:Sequence[PhSel]=None, divs:Sequence[np.ndarray[np.uint16]]=None
                     )->dict[str:np.ndarray[np.ndarray]]:
        """
        Get divisor based inputs for |H2MM| optimization.

        Parameters
        ----------
        origin : PhotonDataS
            Data from which to sort photons.
        statepath : Param, optional
            Definition of streams, if used cannot use bursts or streams kwargs. 
            The default is None.
        bursts : Param, optional
            bursts definition, must be used with streams, and . The default is None.
        streams : Sequence[PhSel], optional
            Streams to inlcude in |H2MM| processing. The default is None.
        divs : Sequence[np.ndarray[np.uint16]], optional
            Per stream divisor, specified in order of streams, 1 array per stream. 
            Each element indicates a divisor threshold.
            The default is None.

        Returns
        -------
        dict[str:np.ndarray[np.ndarray]
             Contains at least the folling keys:
                 indexes : np.ndarray[np.ndarray[np.uint8]]
                     Photon indexes for |H2MM| processing.
        
                times : np.ndarray[np.ndarray[np.int64]]
                    Photon arrival times for |H2MM| processing

        """
        return cls._sort_photons(origin, statepath=statepath, bursts=bursts, streams=streams, divs=divs)

    @classmethod
    def get_ndet(cls, param:Param)->int:
        """
        Get the number of detector indexes of a ntdivStatePath 
        :class:`Param <smfbursts.datamodel.tables.Param>`
        """
        return sum(div.size + 1 for div in param.params['divs'])
        

    @parammethod(origin_as_kw=True)
    def model_streams(cls, statepath:Param, phsel:PhSel, origin:PhotonDataS=None, strict:bool=True)->np.ndarray[np.int64]:
        """
        Determine index(es) of ``phsel`` in the :class:`hm.h2mm_model` based on
        a given param definition.


        Parameters
        ----------
        statepath : Param
            :class:`smfbursts.datamodel.tables.Param` of type :class:`smfburts.datamodel.tables.StatePath` with model.
        phsel : PhSel
            Stream selection to map to model.
        origin : PhotonData, optional
            If specified, the :class:`smfbursts.photondata.PhotonData` object
            from which ``statepath`` is assumed to have been optimized. 
            The default is None.
        strict : bool, optional
            Whether to check if the ``phsel`` contains only streams specified
            in the streams param of ``statepath``. The default is True.
        
        Raises
        ------
        ValueError
            ``phsel`` contains streams not used in ``statepath``.

        Returns
        -------
        used_streams : np.ndarray[np.int64]
            Array of indexes which contribute to ``phsel`` in ``statepath.model.obs``.
            ``statepath.model.obs[:,used_strams].sum(axis=1)`` will return the
            probability that a photon arises from ``phsel`` per state in the 
            ``statepath.model``.

        """
        shift = 0
        indexes = list()
        if strict and phsel - phsel_union(*statepath.params['streams']):
            raise ValueError(f"PhSel {phsel} larger than specified by streams")
        for stream, div in zip(statepath.params['streams'], statepath.params['divs']):
            if stream in phsel:
                if strict and stream - phsel:
                    raise ValueError("")
                indexes += list(range(shift, shift+div.size+1))
            shift += div.size + 1
        return np.array(indexes, dtype=np.int8)

    @classmethod
    def _get_model_lifetime(cls, col:Column, statepath:Param, strict:bool=True, origin:PhotonDataS=None)->np.ndarray[np.float64]:
        r"""
        Compute lifetime of stream in column or param based on model, 
        requires irf_threshold and TCSPC parameters from origin.
        :math:`\tau = \Delta t_{0} / \ln(p_{0}/p_{1} + 1)`
        """
        phsel = col.keytup[0]
        model = statepath.params['model'].obs
        streams = statepath.params['streams']
        divs = statepath.params['divs']
        div_shift = 0 # start of range of h2mm indexes for the given PhSel
        # identify stream and divisor in which phsel occurs
        for i, (stream, div) in enumerate(zip(streams, divs)):
            if phsel in stream:
                if strict and phsel != stream:
                    raise ValueError(f"model optimized for superset of {phsel}")
                if div.size == 0:
                    raise ValueError(f"model optimized without lifetime differentiation in channel {stream}")
                break
            div_shift += div.size + 1
        # get irf threshold, used to identify which div to start at
        if isinstance(origin, PhotonDataList):
            origin = origin.datas[0]
        thresh = origin.irf_thresh.get(stream)
        if thresh is None:
            threshs = [th for tsel, th in origin.irf_thresh.items() if tsel in stream]
            if not threshs:
                if strict:
                    raise ValueError(f"No irf_threshold set for {phsel}")
                warnings.warn("No irf threshold set, lifetimes likely underestimated")
                thresh = 0
            else:
                thresh = max(threshs)
                if all(thresh != th for th in threshs):
                    if strict:
                        raise ValueError("Cannot compute expected lifetime with varied irf_thresholds")
                    warnings.warn("Inconsistent irf_thresholds, using max")
        # locate div to start at 
        elm = list(phsel.ex.elements)[0] if phsel.ex.kind else 0
        div = np.concatenate([[origin.setup.ex_ranges[elm][0,0]], div])
        loc = np.argwhere(thresh < div).reshape(-1)[:2]
        if loc.size == 0:
            loc = np.argwhere(thresh <= div).reshape(-1)[:2]
        if loc.size == 0:
            raise ValueError("no divs greater than irf threshold, cannot compute estimated lifetime")
        elif loc.size == 1:
            if strict:
                raise ValueError("only 1 div range fully after threshold, cannot compute estimated lifetime")
            loc = np.concatenate([loc-1, loc])
        dt = div[loc[1]] - div[loc[0]]
        # equation: lt = deltat_0 / ln(p_0/p_1 + 1)
        c = model[:,div_shift+loc[0]] / model[:,div_shift+loc[1]:].sum(axis=1)
        ex = origin.detdef.get_stream_ids(phsel)[0] // origin.detdef.ex_stride
        lt =  dt / np.log(c + 1) * origin.setup['tcspc_unit'][ex]
        return lt 


###############################################################################
################ Functions/classes for processing usALEX data  ################
###############################################################################

# type hint for following functions for applying shifts to photons (inplace)
TimeShiftFunc = Callable[[np.ndarray[np.int64],np.ndarray[np.int64],np.ndarray[np.bool_],int,...],None]

# shifting for shift with split is much more complicated
def time_shift_c(times:np.ndarray[np.int64], timesper:np.ndarray[np.int64], timesmod:np.ndarray[np.int64], 
                 mask:np.ndarray[np.bool_], ex_ref:int, ex_start:int, scale:float)->None:
    r"""
    Apply the contiguous shift- style shift to photons in times. 
    Note that shift is applied inplace in times array.

    Parameters
    ----------
    times : np.ndarray[np.int64]
        Unshifted photon times.
    timesper : np.ndarray[np.int64]
        Time of beginning of alex period of each photon.
    timesmod : np.ndarray[np.int64]
        Time since beginning of alex period of each photon.
    mask : np.ndarray[np.bool\_]
        Photons to shift (matching det).
    ex_ref : int
        Start of excitation period of source.
    ex_start : int
        Start of destination excitation period.
    scale : float
        Ratio of duration of destination excitation period to source excitation period.

    """
    times[mask] = timesper[mask] + (scale*(timesmod[mask]-ex_ref)).astype(np.int64) + ex_start


def time_shift_n(times:np.ndarray[np.int64], timesper:np.ndarray[np.int64], timesmod:np.ndarray[np.int64], 
                 mask:np.ndarray[np.bool_], ex_refb:int, ex_startb:int, scaleb:float, 
                 ex_reff:int, ex_startf:int, scalef:float)->None:
    r"""
    Apply the nearest shift- style shift to photons in times. 
    Note that shift is applied inplace in times array.

    Parameters
    ----------
    times : np.ndarray[np.int64]
        Unshifted photon times.
    timesper : np.ndarray[np.int64]
        Time of beginning of alex period of each photon.
    timesmod : np.ndarray[np.int64]
        Time since beginning of alex period of each photon.
    mask : np.ndarray[np.bool\_]
        Time since beginning of alex period of each photon.
    ex_refb : int
        Time of beginning of back shift source excitation period.
    ex_startb : int
        Time of beginning of previous destination excitation period subsection.
    scaleb : float
        Ratio of size of previous destination excitation period subsection to 
        source excitation period subsection before division.
    ex_reff : int
        Time in source excitation period to split between backward and forward shift.
    ex_startf : int
        Time of beginnig of next destination excitation period subsection.
    scalef : float
        Ratio of size of next destination excitation period subsection to source
        excitation periods subsection after division.

    """
    maskt = mask & (timesmod < ex_reff)
    times[maskt] = timesper[maskt] + (scaleb*(timesmod[maskt]-ex_refb)).astype(np.int64) + ex_startb
    maskt = mask & (timesmod >= ex_reff)
    times[maskt] = timesper[maskt] + (scalef*(timesmod[maskt]-ex_reff)).astype(np.int64) + ex_startf


def time_even_c(times:np.ndarray[np.int64], timesper:np.ndarray[np.int64], timesmod:np.ndarray[np.int64], 
                mask:np.ndarray[np.bool_], ex_ref:int, ex_start:int, ex_end:int)->None:
    r"""
    Apply the contiguous even- style shift to photons in times
    Note that shift is applied inplace in times array.

    Parameters
    ----------
    times : np.ndarray[np.int64]
        Unshifted photon times.
    timesper : np.ndarray[np.int64]
        Time of beginning of alex period of each photon.
    timesmod : np.ndarray[np.int64]
        Time since beginning of alex period of each photon.
    mask : np.ndarray[np.bool\_]
        Photons to shift (matching det).
    ex_ref : int
        Start of excitation period of source.
    ex_start : int
        Start of destination excitation period.
    ex_end : int
        Ratio of duration of destination excitation period to source excitation period.

    """
    tper, cnts = np.unique(timesper[mask], return_counts=True)
    for t, c in zip(tper, cnts):
        maskt = (timesper == t) & mask
        times[maskt] = np.linspace(ex_start, ex_end, c, dtype=np.int64) + t


def time_even_n(times:np.ndarray[np.int64], timesper:np.ndarray[np.int64], timesmod:np.ndarray[np.int64], 
                mask:np.ndarray[np.bool_], thresh:int, ex_startr:int, ex_endr:int, ex_startf:int, ex_endf:int)->None:
    r"""
    Apply the nearest even- style shift to photons in times
    Note that shift is applied inplace in times array.

    Parameters
    ----------
    times : np.ndarray[np.int64]
        Unshifted photon times.
    timesper : np.ndarray[np.int64]
        Time of beginning of alex period of each photon.
    timesmod : np.ndarray[np.int64]
        Time since beginning of alex period of each photon.
    mask : np.ndarray[np.bool\_]
        Photons to shift (matching det).
    thresh : int
        Time of division between photons to shift backwards vs forwards.
    ex_startr : int
        Start time of destination period for shifting backwards.
    ex_endr : int
        End time of destination period for shifting backwards.
    ex_startf : int
        Start time of destination period for shifting forwards.
    ex_endf : int
        End time of desintation period for shifting forwards.

    """
    tper = np.unique(timesper[mask])
    for t in tper:
        maskt = (timesper == t) & mask
        masktt = maskt & (timesmod < thresh)
        times[masktt] = np.linspace(ex_startr, ex_endr, masktt.sum(), dtype=np.int64) + t
        masktt = maskt & (timesmod >= thresh)
        times[masktt] = np.linspace(ex_startf, ex_endf, masktt.sum(), dtype=np.int64) + t


def time_rand_c(times:np.ndarray[np.int64], timesper:np.ndarray[np.int64], timesmod:np.ndarray[np.int64], 
                mask:np.ndarray[np.bool_], ex_start:int, ex_end:int, gen:np.random.Generator)->None:
    r"""
    Apply continguous random style shift to photons in times.
    Note that the shift is applied inplace in the times array.

    Parameters
    ----------
    times : np.ndarray[np.int64]
        Unshifted photon times.
    timesper : np.ndarray[np.int64]
        Time of beginning of alex period of each photon.
    timesmod : np.ndarray[np.int64]
        Time since beginning of alex period of each photon.
    mask : np.ndarray[np.bool\_]
        Photons to shift (matching det).
    ex_start : int
        Start of destination excitation period.
    ex_end : int
        Stop of destination excitation period.
    gen : np.random.Generator
        random number generator.

    """
    tper, cnts = np.unique(timesper[mask], return_counts=True)
    for t, c in zip(tper, cnts):
        maskt = (timesper==t)&mask
        times[maskt] = np.sort(gen.integers(ex_start, ex_end, c))
        times[maskt] += timesper[maskt]


def time_rand_n(times:np.ndarray[np.int64], timesper:np.ndarray[np.int64], timesmod:np.ndarray[np.int64], 
                mask:np.ndarray[np.bool_], thresh:int, 
                ex_startr:int, ex_endr:int, ex_startf:int, ex_endf:int, gen:np.random.Generator)->None:
    r"""
    Apply nearest random style shift to photons in times.
    Note that the shift if applied inplace in the times array.

    Parameters
    ----------
    times : np.ndarray[np.int64]
        Unshifted photon times.
    timesper : np.ndarray[np.int64]
        Time of beginning of alex period of each photon.
    timesmod : np.ndarray[np.int64]
        Time since beginning of alex period of each photon.
    mask : np.ndarray[np.bool\_]
        Photons to shift (matching det).
    thresh : int
        Time of division between photons to shift backwards vs forwards.
    ex_startr : int
        Start time of destination period for shifting backwards.
    ex_endr : int
        End time of destination period for shifting backwards.
    ex_startf : int
        Start time of destination period for shifting forwards.
    ex_endf : int
        End time of desintation period for shifting forwards.
    gen : np.random.Generator
        random number generator.

    """
    tper = np.unique(timesper[mask])
    for t in tper:
        maskt = (timesper == t) & mask
        masktt = maskt & (timesmod < thresh)
        times[masktt] = np.sort(gen.integers(ex_startr, ex_endr, masktt.sum()))
        times[masktt] += timesper[masktt]
        masktt = maskt & (timesmod >= thresh)
        times[masktt] = np.sort(gen.integers(ex_startf, ex_endf, masktt.sum()))
        times[masktt] += timesper[masktt]


def _get_ex_ranges(setup:PhSpec, source_ex:int, dest_ex:int)->tuple[int,int,int,int,int]:
    """
    Get excitation ranges from setup, defined by source_ex and dest_ex, 
    returns as source_min, source_max, dest_min, dest_max, period.
    Period is the alex_period, specific for usALEX experiments.
    """
    ex_rng, period = setup['ex_ranges'], setup['alex_period']
    if ex_rng[source_ex].shape[0] != 1 or ex_rng[dest_ex].shape[0] != 1:
        raise ValueError("shift for fragmented excitation ranges not supported")
    source_min, source_max = ex_rng[source_ex][0,0], ex_rng[source_ex][0,1]
    dest_min, dest_max = ex_rng[dest_ex][0,0], ex_rng[dest_ex][0,1]
    return source_min, source_max, dest_min, dest_max, period


### Functions for converting excitation range to shift into inputs for time function calls
def _shift_cshiftreg(source_min:int, source_max:int, dest_min:int, dest_max:int, gen:np.random.Generator):
    """split function for shift splitting, returns tuple and remaining elements are args for func"""
    return time_shift_c, source_min, dest_min, (dest_max-dest_min)/(source_max-source_min)


def _shift_cevenreg(source_min:int, source_max:int, dest_min:int, dest_max:int, gen:np.random.Generator):
    """split function for even splitting, returns tuple and remaining elements are args for func"""
    return time_even_c, source_min, dest_min, dest_max


def _shift_crandreg(source_min:int, source_max:int, dest_min:int, dest_max:int, gen:np.random.Generator):
    """split function for rand shifting, returns tuple and remaining elements are args for func"""
    return time_rand_c, dest_min, dest_max, gen


def _shift_nshiftreg(source_min:int, source_max:int, split, dest_minr:int, dest_maxr:int, dest_minf:int, dest_maxf:int, gen:np.random.Generator):
    """split function for shift splitting, returns tuple and remaining elements are args for func"""
    scaler = (dest_maxr-dest_minr)/(split-source_min)
    scalef = (dest_maxf-dest_minf)/(source_max-split)
    return time_shift_n, source_min, dest_minr, scaler, split, dest_minf, scalef


def _shift_nevenreg(source_min, source_max, split, dest_minr, dest_maxr, dest_minf, dest_maxf, gen):
    """Split function for even splitting, returns tuple and remaining elements are args for func"""
    return time_even_n, split, dest_minr, dest_maxr, dest_minf, dest_maxf


def _shift_nrandreg(source_min, source_max, split, dest_minr, dest_maxr, dest_minf, dest_maxf, gen):
    """Split function for rand splitting, returns tuple and remaining elements are args for func"""
    return time_rand_n, split, dest_minr, dest_maxr, dest_minf, dest_maxf, gen


# dictionaries for string to shift func conversion
_csplit_regfuncs = {'shift':_shift_cshiftreg, 'even':_shift_cevenreg, 'rand':_shift_crandreg}
_nsplit_regfuncs = {'shift':_shift_nshiftreg, 'even':_shift_nevenreg, 'rand':_shift_nrandreg}


# Functions for processing excitation ranges by selected shift pattern
def _shift_ex_continguous(setup:PhSpec, source:int, dest:int, sort:str, 
                         gen:np.random.Generator)->tuple[TimeShiftFunc, int,...]:
    source_min, source_max, dest_min, dest_max, alexper = _get_ex_ranges(setup, source, dest)
    """Generate shifting function for contiguous style shifts"""
    ds = abs(dest_min - source_min)
    df = alexper + dest_min - source_min
    dr = source_min + alexper - dest_min
    if ds < df and ds < dr:
        return _csplit_regfuncs[sort](source_min, source_max, dest_min, dest_max, gen)
    elif dr < dr:
        return _csplit_regfuncs[sort](source_min, source_max, dest_min-alexper, dest_max-alexper, gen)
    return _csplit_regfuncs[sort](source_min, source_max, dest_min+alexper, dest_max+alexper, gen)


def _shift_threshold(s0, s1, d0, d1, p)->tuple[int,int]:
    """
    Compute limits of shift, 
    s0, s1 are source limits, d0, d1 destinstion limits, p alex period
    returns limits (start, stop) inside [s0, s1]
    """
    pp = p/2
    st = ((d1*s1)-(d0*s0)+(s0-s1)*pp)/(d1-d0+s1-s0)
    return st, st + pp


def _shift_ex_nearest(setup:PhSpec, source:int, dest:int, sort:str, 
                     gen:np.random.Generator)->tuple[TimeShiftFunc,int,...]:
    """Generate nearest shift function, returns tuple and remaining elements are args for func """
    s0, s1, d0, d1, p = _get_ex_ranges(setup, source, dest)
    if s0 < d0:
        st, dt = _shift_threshold(s0, s1, d0, d1, p)
        dminr, dmaxr, dminf, dmaxf = dt-p, d1-p, d0, dt
    else:
        dt, st = _shift_threshold(d0, d1, s0, s1, p)
        dminr, dmaxr, dminf, dmaxf = dt, d1, d0+p, dt+p
    if st < s0:
        return _csplit_regfuncs[sort](s0, s1, d0, d1, gen)
    return _nsplit_regfuncs[sort](s0, s1, st, dminr, dmaxr, dminf, dmaxf, gen)

# examples of valid shift strings:
# base, "neven:0", "ceven:1", "nshift:1", "cshift:2", "nrand0xfa0:2", "crand0x3cge:0"
_shift_regex = re.compile(r'base|((?P<way>n|c)(?P<sort>shift|even|(?P<rand>rand))(?(rand)0x(?P<seed>[\da-f]{1,16})|)\:\w?(?P<ex>\d+))')


def _proc_shift(setup:PhSpec, phsel:PhSel, shift:str, gens:dict[int,np.random.Generator]):
    """Get shifting function for single excitation/detector"""
    det_ids = setup['detdef'].get_stream_ids(phsel)
    ex_ids = det_ids % setup['detdef'].ex_stride
    shiftm = _shift_regex.fullmatch(shift)
    if shiftm is None:
        raise ValueError(f"'{shift}' is invalid shift specification")
    if shiftm.group('sort') is None:
        return tuple()
    if shiftm.group('ex'):
        seed = int(shiftm.group('ex'),16)
        if seed in gens:
            gen = gens[seed]
        else:
            gen = np.random.default_rng(seed)
            gens[seed] = gen
    else:
        gen = None
    dest = int(shiftm.group('ex'))
    func = _shift_ex_nearest if shiftm.group('way') == 'n' else _shift_ex_continguous
    return ((d_id, func(setup, ex_id, dest, shiftm.group('sort'), gen))
            for d_id, ex_id in zip(det_ids, ex_ids))


def _proc_shifts(setup:PhSpec, phsels:Sequence[PhSel], shifts:Sequence[str], gens:dict[int:np.random.Generator])->tuple:
    """Interpret the shifts param for :class:`usAlexStatePath`"""
    if len(phsels) != len(shifts):
        raise ValueError("must specify same number of streams as shifts")
    if any(not isinstance(phsel, (PhSel,PhStream)) for phsel in phsels):
        raise TypeError("all elements of phsels must be PhSel")
    return tuple(chain.from_iterable(_proc_shift(setup, phsel, shift, gens) for phsel, shift in zip(phsels, shifts)))


def _validate_shiftstr(shift:str)->str:
    """Validator for shift string code"""
    smtch = _shift_regex.fullmatch(shift)
    if smtch is None:
        raise ValueError("invalid shift string")
    if smtch.group('sort') is None:
        return 'base'
    out = smtch.group('way')+smtch.group('sort')
    if smtch.group('seed'):
        out += '0x'+ format(int(smtch.group('seed'), 16), '016x')
    out += ":" + smtch.group('ex').lstrip('0')
    out += '0' if out[-1] == ':' else '' # in case ex == 0 and lstrip removes all
    return out


def sort_usALEX_times_indexes(table:BasePhotonTable, phsels:Sequence[PhSel], shifts:Sequence[str]
              )->tuple[np.ndarray[np.ndarray[np.int8]],np.ndarray[np.ndarray[np.uint64]],np.ndarray[np.ndarray[np.int64]]]:
    """
    Generate time-shifted times, indexes, and sort order arrays for 
    |H2MM| analysis of usALEX data. 
    This function is used by :class:`usAlexStatePath` to generate the underlying
    mapping of raw data to |H2MM|

    Parameters
    ----------
    table : BasePhotonTable
        ``BasePhotonTable`` that is the source of the burst data 
        (defines time ranges to analyze).
    phsels : Sequence[PhSel]
        Defines the phsel->|H2MM| index map, ie [phsel0, phsel1] map to [0, 1]
        in |H2MM| index
    shifts : Sequence[str]
        Definition for how to shift each index in phsels, this is the .

    Returns
    -------
    out_idx : np.ndarray[np.ndarray[np.int8]]
        Time shifted indexes for use in |H2MM| analysis.
    out_times : np.ndarray[np.ndarray[np.int64]]
        Time shifted times for use in |H2MM| analysis.
    out_sort : np.ndarray[np.ndarray[np.int64]]
        Map from |H2MM| shifted order to original order, 
        ie out_idx[out_sort] = original_order_idx.

    """
    origin = table.origin
    setup = origin.setup
    gens = dict()
    shift_spec = _proc_shifts(setup, phsels, shifts, gens)
    idx_map = reindex_phsel(setup['detdef'], phsels)
    out_times = np.empty(table.size, dtype=np.object_)
    out_idx = np.empty(table.size, dtype=np.object_)
    out_sort = np.empty(table.size, dtype=np.object_)
    phselu = phsel_union(*phsels)
    alexper, alexoff = setup['alex_period'], setup['alex_offset']
    for i, (time, det) in enumerate(zip(table.iter_column('ph_times', phselu),
                                          table.iter_column('ph_dets', phselu))):
        time, idx = time.copy(), idx_map[det]
        for d_id, specs in shift_spec:
            func, args = specs[0], specs[1:]
            tper, tmod = divmod(time-alexoff, alexper)
            tper = tper*alexper + alexoff
            func(time, tper, tmod, det == d_id, *args)
        sort = np.argsort(time)
        out_times[i] = time[sort]
        out_idx[i] = idx[sort].astype(np.uint8)
        out_sort[i] = np.argsort(sort)
    return out_idx, out_times, out_sort


class usAlexStatePath(StatePath):
    r"""
    *Viterbi* StatePath for |H2MM| analysis where certain streams are shifted
    according to several available schemes so that excitiation periods overlap.
    
    This is used primarily to implement the method in 
    `Harris et. al. 2022 <https://doi.org/10.1038/s41467-022-28632-x>`_ for dealing
    with :math:`\mu s` ALEX measurements.
    
    The parameters are the same as :class:`StatePath` with the addition of the 
    ``shifts`` parameter. This is a tuple of the same length as ``streams`` which
    defines the function 
    
    For non-shifted streams, the code should be "base"
    For shifted photon streams, the string specifies 3 pieces of information
    
    1. Direction of shift options are "n" and "c" for nearest, and contiguous, respectively.
    2. Style of shift, may be "shift", "even" or "rand0x<hex>"
    3. Destination excitation period (an integer).
    
    These are assembled as follows ``<direction><style>:<destination>``
    So for a contiguous, even shift into excitation period 0, the string would
    be ``"ceven:0"``.
    
    Regarding the **direction** of the shift
    
    - "c" (continguous) shift results in all photons from a given excitation
      period will be shifted into the same destination excitation period
    - "n" (nearest) shift results in photons being shifted to the temporally
      nearest destination excitation period.
    
    Regarding the styles of shifts
    
    - "shift" is the simplest shifting technique, where an apporpirate linear
      shift is applied to each photon, so that spacing of shifted photons is 
      proportionally not changed 
      (a scaling factor is necessary to account for differences in the size of excitation periods)
    - "even" results in the even distribution of shifted photons into their destination
      period.
    - "rand0x<hex>" note that "<hex>" shoudl be replaced by a hexidecimal specification.
      Photon time are randomly distributed in the destination period.
      "<hex>" value is used as the seed for the random number generator generating
      the times. So long as numpy version is the same, this ensures reproducability
      of results. A random shift into excitation period 0 and using the nearest
      excitation period and a seed of ``0x2f4c`` the string would be
      ``"nrand0x2f4c:0"``.
    
    Params
    ------
        model : hm.h2mm_model
            The :class:`hm.h2mm_model` used in *Viterbi* processing.
        streams : tuple[PhSel, ...]
            tuple of :class:`PhSel` defining the indexes of photons in |H2MM| processing.
        shifts : tuple[str, ...]
            String describing how to shift (which shift function to use) the given
            stream, tuple must be same length as streams.
        
    Parents
    -------
    Note that these are the same as :class:`StatePath`
    
        bursts : Param[BasePhotonTable]
            Usually a :class:smfbursts.datamodel.tables.Bursts` :class:`smfbursts.datamodel.tables.Param`, defining the time 
            ranges of each "burst" in |H2MM| processing.
    
    Columns
    -------
    Note that these are the same as :class:`StatePath`
    
        indexpath : np.ndarray[np.uint8], ()
            Actual indexes used in |H2MM| processing. Each row is uint8 array.
        detpath : np.ndarray[np.uint8], ()
            Detector indexes of photons used in |H2MM| processing, not reassigned by streams.
            Each row is uint8 array.
        timepath : np.ndarray[np.int64], ()
            Actual times used in |H2MM| processing. Each row is int64 array.
        statepath : np.ndarray[np.uint8], ()
            Most likely states of each photon as processed in *Viterbi* algorithm.
            This is the direct output of *Viterbi*, no sub-selecting photons etc.
            Each row is 1d uint8 array
        scalepath : np.ndarray[np.float64], ()
            Posterior per photon of *Viterbi* processing. This is direct output of *Viterbi*.
            Each row is 1D float64 array.
        pathllpath : np.ndarray[np.float64], ()
            Log-likelihood of most likely state of each photon. This is direct output of *Viterbi*.
            Each row is 1D float64 array.
        gammapath : np.ndarray[np.float64], ()
            Gamma array, giving likelihood per-photon per-state.
            Each row is 2d float64 array, indexed [photon, state].
        ph_index : np.ndarray[np.ndarray[np.int8]], (phsel:PhSel, )
            Indexes of photons used in |H2MM| processing, mappped/masked by phsel.
            Photons ouside of model-streams receive a value of -1.
            Each row is 1D int8.
        ph_h2mmtime : np.ndarray[np.ndarray[np.int64]], (phsel:PhSel, )
            Times of photons used in |H2MM| processing mapped/masked by phsel. 
            Photons outside of model-streams receive value of -1. 
            Rows are 1d int64 arrays.
        ph_state : np.ndarray[np.ndarray[np.int8]], (phsel:PhSel, )
            *Viterbi* state of each photon, mapped/masked by phsel. 
            Photons outside of model-streams receive value of -1. 
            Rows are 1d int8 arrays.
        ph_scale : np.ndarray[np.ndarray[np.float64]], (phsel:PhSel, )
            Posterior likelihood of each phton, mapped/masked by phsel.
            Photons outside of model-streams receive value of nan.
            Rows are 1d float64 arrays.
        ph_ll : np.ndarray[np.ndarray[np.float64]], (phsel:PhSel, )
            log-likihood of state-assignment of photon for *Viterbi* path, mapped/masked by phsel.
            Photons outside of model-streams receive value of nan.
            Rows are 1d float64 arrays
        ph_gamma : np.ndarray[np.ndarray[np.float64]], (phsel:PhSel, )
            Gamma array, giving likelihood per-photon per-state mapped/masked by phsel.
            Each row is 2d float64 array, indexed [photon, state].
        bstates: ()
            Bitcode indicating which states present in burst. If given bit position
            is present, then that state is present in burst, ie if states 0 and 2 are
            in a given burst, the the value is 0b00000101 = 5. Rows are int.
        eff_state : (phsel:PhSel, )
            "Effective" state of each photon in burst maped/masked by phsel. 
            If a photon is in a stream not present in model streams, then infer state 
            by nearest photon that is in streams.
            This is essentially ph_state with -1s replaced with an infered state.
            Rows are 1d int8 arrays.
    
    """
    #: :meta private:
    param_defs = StatePath.param_defs + (
        ParamDef('shifts', TV_tuple(typedefs=TV_str(validator=_validate_shiftstr))),
        )
    #: :meta private:
    column_defs = h2mm_columndefs + (
        ColumnDef('sortpath', tuple(), 0, 'all', get_func='_get_sortpath', dtype=np.object_, typedef=np.dtype('<i8')),
        )
    _sort_store = ('sort', )

    @classmethod
    def param_preprocess(cls, param:Sequence[tuple[str,Any]]|tupledict, parents:dict[str,Param])->tuple[dict,dict]:
        """:meta private:"""
        param = as_paramdict(param, tuple(pdef.name for pdef in cls.param_defs)+('sort',))
        if isinstance(parents, Param):
            parents = {'bursts':parents}
        elif isinstance(parents, tupledict):
            parents = parents.asdict
        if 'bursts' not in parents:
            raise ValueError("Must define bursts parent")
        if 'shifts' in param:
            return param, parents
        streams, detdef = param['streams'], parents['bursts'].detdef
        sort = param.pop('sort', 'neven')
        sort = sort+f'0x{np.random.randint(0,1<<63):x}' if 'rand' in sort and '0x' not in sort else sort
        streams = tuple(stream.render_positive(detdef, convert_all=True) for stream in streams)
        if len(streams[0].ex.elements) != 1:
            raise ValueError("assumed base stream has multiple excitations")
        exs = {stream.ex for stream in streams}
        if len(exs) == 1:
            raise ValueError('All streams have same excitation, cannot infer shift')
        base_ex = streams[0].ex
        dest = str(list(base_ex)[0])
        shifts = tuple(f'{sort}:{dest}' if stream.ex != base_ex else stream for stream in streams)
        param['streams'] = streams
        param['shifts'] = ('base', ) + shifts
        return param, parents

    @cite('HarrisNatComms2022', purpose='usALEX photon shift for multiparamter H2MM analysis')
    def __init_columns__(self):
        super().__init_columns__()

    @classmethod
    def validate_param(cls, param:Param):
        """:meta private:"""
        detdef = param.detdef
        nstream = _validate_model_streams(param.params['streams'], detdef)
        if (ndet:=param.params['model'].ndet) != nstream:
            raise ValueError(f"Mismatched model to number of dets, defined {ndet} detectors but model has {ndet}")
        shifts = param.params['shifts']
        if len(shifts) != nstream:
            raise ValueError(f'divs must have same number of elements as streams, got {nstream} and {len(shifts)}')
        if any(((ex:=int(_shift_regex.match(shift).group('ex')))) >= detdef.ex for shift in shifts if shift != 'base'):
            raise ValueError(f"destination ex window out of range: {ex} for size of {detdef.ex}")
        if all(s != 'base' for s in shifts):
            raise ValueError("no shifts applied, use StatePath instead")

    @classmethod
    def _sort_photons_func(cls, origin:PhotonDataS, bursts, streams:Sequence[PhSel], shifts:Sequence[str]):
        indexes, times, sort = sort_usALEX_times_indexes(origin.get_table(bursts), streams, shifts)
        return dict(indexes=indexes, times=times, sort=sort)

    def _get_sortpath(self):
        """
        Retrieve the sortpath for photons of the current table, 
        allows mapping from |H2MM| order to original
        """
        return self._sort_photons(self.origin, statepath=self.param)['sort']

    def phsel_select(self, phsel:PhSel, col:str, fill:Any, dtype:np.dtype)->np.ndarray[np.object_]:
        r"""
        Maps the of the inputs to |H2MM| evaluation to the "unprocessed" output shape.

        Parameters
        ----------
        phsel : PhSel
            A phsel object defining the output streams to return.
        col : str
            Name of column being returned.
        fill : Any
            Value to fill any photons that are in phsel but outside of phsel_span.
        dtype : np.dtype
            Data-type of output array.

        Returns
        -------
        np.ndarray[np.object\_]
            If implemented should return object array of column maped to phsel

        """
        out = np.empty(self.size, dtype=np.object_)
        phselall = self.phsel_span
        for i, (arr, ms, md, sort) in enumerate(zip(self.iter_column(col), 
                                                   self.base_table.iter_column('ph_mask', phselall),
                                                   self.base_table.iter_column('ph_mask', phsel)),
                                                   self.iter_column('sortpath')):
            out[i] = _mask_expand(arr[sort], ms, md, fill)
        return out


class StatePathFilter(BasePhotonTable):
    """
    Bursts with state(s) removed. Motivated by |Kache2005|.
    This BasePhotonTable defines the contiguous time periods where:
    
        - The photons are defined to be "in burst" as defined by the burst defintion
          of the parent StatePath
        - The system is not in the excluded state(s).
    
    The result of this is that bursts starting or ending with the excluded state(s)
    are truncated, and bursts with a transition into and out of the excldued state(s)
    are split into two or more bursts with the transitions into the excludes state(s)
    removed.
    
    Parents
    -------
    statepath : StatePath
        Defines the model and bursts by which photon states are defined.
    
    Params
    ------
    exclude: frozenset[int]
        All states to be ecluded 
    
    """
    #: :meta private:
    param_defs = (ParamDef('exclude', TV_frozenset(typedefs=TV_int(mn=0, ), minsize=1)), )
    #: :meta private:
    parent_defs = (ParentDef('statepath', StatePathBase), )
    #: :meta private:
    column_defs = make_base_column_defs()

    @cite("KacheBMCMeth2025", purpose="Trimming bursts by Viterbi state")
    def __init_columns__(self):
        statepath = self.parents['statepath']
        bursts = statepath.parents['bursts']
        starts, stops = list(), list()
        exclude = np.array(list(self.param.params['exclude']))
        for bstart, bstop, times, states in zip(bursts.iter_column('start'),
                                                bursts.iter_column('stop'),
                                                statepath.iter_column('timepath'), 
                                                statepath.iter_column('statepath')):
            mask = np.isin(states, exclude) # sort photon by in/out of exclude
            tdiff = np.diff(mask.astype(np.int8)) # mask transitions, -1 means transition into burst, 1 transition out of burst
            inmask = np.argwhere(tdiff < 0)[:,0] # indexes of end of transition into burst
            outmask = np.argwhere(tdiff > 0)[:,0] # indexes of end of transition out of burst
            startbeg, startend = times[inmask], times[inmask+1] # get last photon before and first photon after beginning of burst
            stopbeg, stopend = times[outmask], times[outmask+1] # last photon before and first photon after end of burst
            start = startbeg + ((startend - startbeg) // 2) # compute midmpoint of start of burst
            stop = stopbeg + ((stopend - stopbeg) // 2) # midpoints of transition into/out of burs
            if not mask[0]:
                starts.append([bstart])
            starts.append(start)
            stops.append(stop)
            if not mask[-1]:
                stops.append([bstop])            
        starts = np.concatenate(starts).astype('<i8')
        stops = np.concatenate(stops).astype('<i8')
        self._add_column('start', tuple(), starts)
        self._add_column('stop', tuple(), stops)
        istart, istop = smc.index_ranges(self.origin.times, starts, stops)
        self._add_column('istart', tuple(), istart)
        self._add_column('istop', tuple(), istop)

    @classmethod
    def _validate_param(cls, param:Param)->None:
        if max(param.params['exclude']) >= param.parents['statepath'].params['model'].nstate:
            raise ValueError("Cannot exclude states larger than number of states in model")

    @paramproperty
    def detdef(self, param:Param)->DetDef:
        """paramproperty that gives the ``DetDef`` of the current ``Param``"""
        return param.parents['statepath'].detdef

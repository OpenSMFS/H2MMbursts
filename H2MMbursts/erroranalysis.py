#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Paul David Harris
# Created:  05/01/2026
# Purpose: Analysis of Error in H2MM
"""
Error Analysis
==============

This module prodvides functions for assessing the error of |H2MM| optimizations.

There are 2 primary methods of computing the error
#. Decrease in loglikelihood, handeled by :func:`statepath_ll_error`
#. Bootstrap, handled by :class:`BootStrapError`

Decrease in loglikelihood computes the error by varying one parameter and locating
the point at which the loglikelihood decreases by a certain amount (typically 0.5).

Bootstrap computes the error using the variance of the optimized models of subsets.


.. |H2MM| replace:: H:sup:`2` MM
.. |StatePath| replace:: :class:`bhm.StatePathBase <H2MMbursts.modeltables.StatePath>`
.. |scipyoptimize| replace:: `scipy.optimize.fminbound <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fminbound.html>`__
"""
from typing import Literal
from collections.abc import Sequence, Callable
from numbers import Number
from functools import partial
from itertools import product, repeat

import numpy as np
from scipy import optimize

import H2MM_C as hm
from smfbursts.datamodel.utils import tupledict, ImDict, _dict_update
from smfbursts.datamodel.immutabledata import _ImData, TV_tuple, TV_str
from smfbursts.datamodel.tables import Param, Column, TV_Param
from smfbursts.photondata import PhotonDataS

from .modeltables import StatePathBase, TArray, DArray, TV_H2MMModel, Dwells

#: Generic alias for a location sequence, ie input specifying a location in an array (used for type hinting)
LSeq = Sequence[tuple[int]|tuple[int,int]]
#: Generic alias for input to an adjust function specification (used for type hinting)
AFunc = Callable[[hm.h2mm_model,Sequence[float],LSeq],hm.h2mm_model]


def _proc_tupmodels(bse:"BootStrapError", *args, **kwargs):
    return dict(ndet=bse.param.params['model'].ndet, nstate=bse.param.params['model'].nstate)


class BootStrapError(_ImData):
    """
    Class for assessing error using a bootstrap style method of the values of
    an |H2MM| model
    """
    __slots__ = ('dataid', 'param', 'models')
    _typeconversions = ImDict(dataid=TV_str, param=TV_Param(table_type=StatePathBase), models=TV_tuple(typdefs=TV_H2MMModel, data_proc=_proc_tupmodels))
    _required = frozenset({'dataid', 'param', 'models'})
    
    @property
    def tp(self)->type:
        """Type of burst definition of input table"""
        return self.param.tp
    
    @property
    def params(self)->tupledict:
        """Params of statepath to which statepath was assessed"""
        return self.param.params
    
    @property
    def model(self)->int:
        return self.param.params['model']
    
    @property
    def nstate(self)->int:
        """Number of states in mode"""
        return self.param.params['model'].nstate
    
    @property
    def n(self)->int:
        """Number of bootstrap optimizations used to compute error"""
        return len(self.models)
    
    @property
    def std_prior(self)->np.ndarray[np.float64]:
        """Standard deviation of values of prior array"""
        return np.std([m.prior for m in self.models], axis=0)
    
    @property
    def err_prior(self)->np.ndarray[np.float64]:
        """Standard error of values of prior array"""
        return self.std_prior / np.sqrt(self.n)

    @property
    def std_trans(self)->np.ndarray[np.float64]:
        """Standard deviation of values of trans array"""
        return np.std([m.trans for m in self.models], axis=0)
    
    @property
    def err_trans(self)->np.ndarray[np.float64]:
        """Standard error of values of trans array"""
        return self.std_trans /np.sqrt(self.n)
    
    @property
    def std_obs(self)->np.ndarray[np.float64]:
        """Standard deviation of values of obs array"""
        return np.std([m.obs for m in self.models], axis=0)
    
    @property
    def err_obs(self)->np.ndarray[np.float64]:
        """Standard error of values of obs array"""
        return self.std_obs / np.sqrt(self.n)
    
    @classmethod
    def evaluate(cls, origin:PhotonDataS, statepath:Param, n:int=10)->"BootStrapError":
        """
        Evaluate bootstrap error of given :class:`H2MMbursts.modeltables.StatePath` 
        Param.

        Parameters
        ----------
        origin : PhotonDataS
            Data on which error is based.
        statepath : Param
            |StatePath| based Param to assess error.
        n : int, optional
            Number . The default is 10.

        Returns
        -------
        BootStrapError
            Bootstrap error assesment of ``statepath``.

        """
        tp, model = statepath.tp, statepath.params['model']
        arrs = tp._sort_photons(origin, bursts=statepath.base_param, **statepath.params)
        models = tuple(model.optimize(arrs['indexes'][i::n], arrs['times'][i::n], inplace=False) for i in range(n))
        return cls(dataid=origin.dataID, param=statepath, models=models)
    
    def col_std(self, col:Column)->np.ndarray[np.float64]:
        """Compute expected standard deviation of a given column"""
        return np.std([self.param.model_value(col)
                       for model in self.models], axis=0)

    def col_error(self, col:Column)->np.ndarray[np.float64]:
        """Compute expected standard error of given column"""
        return self.col_std(col) / np.sqrt(self.n)


#: Generic alias for type hinting for array location specification.
#: Possible Values are
#: 1. None, indicating the whole array. 
#: 2. a tuple of ints specifying a single location (same number of ints as ndim of target array)
#: 3. a tuple of locations each specified as in 2
#: 4. a boolean array of same shape as target array
LocSpec = None|tuple[int|tuple[int,...],...]|np.ndarray[np.bool_]


def _loc_reshape(loc:LocSpec, shape:tuple[int,...])->np.ndarray[np.bool_]:
    """Return mask of shape shape and True at locations defined by loc"""
    if loc is None:
        return np.ones(shape, dtype=np.bool_)
    loc = np.asarray(loc)
    if loc.dtype == np.bool_:
        if loc.shape != shape:
            raise ValueError('Mismatched boolean shape')
        return loc
    mask = np.zeros(shape, dtype=np.bool_)
    loc = loc.reshape(-1, len(shape))
    for p in loc:
        mask[tuple(p)] = True
    return mask


def prior_adjust(model:hm.h2mm_model, val:float, loc:LocSpec, outer:LocSpec=None)->hm.h2mm_model:
    r"""
    Return a model with prior array "adjusted" by factor val at locations loc,
    if outer specified, only touch outer

    Parameters
    ----------
    model : hm.h2mm_model
        Model to adjust.
    val : float
        Factor by which to increase locations set by loc.
    loc : tuple[int|tuple[int,...],...]|np.ndarray[np.bool\_]
        Locations to adjust, eitehr tuple if indexes of boolean mask of prior array.
        See :attr:`LocSpec` for general format description.
    outer : tuple[int|tuple[int,...],...]|np.ndarray[np.bool\_], optional
        Values cross which to allow change (includes loc). either tuple of locations
        or boolean mask of prior array. See :attr:`LocSpec` for general format description.
        The default is None.

    Raises
    ------
    ValueError
        Bad specification of loc argument.

    Returns
    -------
    hm.h2mm_model
        prior adjusted model.

    """
    prior, trans, obs = model.prior.copy(), model.trans.copy(), model.obs.copy()
    loc = _loc_reshape(loc, prior.shape)
    outer = _loc_reshape(outer, prior.shape)
    inv = outer ^ loc
    if np.any(loc & inv):
        raise ValueError("loc is not subset of outer")
    outsum = prior[outer].sum()
    prior[loc] = val* prior[loc] / prior[loc].sum() * outsum
    prior[inv] = (1-val)* prior[inv] / prior[inv].sum() * outsum
    return hm.h2mm_model(prior, trans, obs)


def _arr_adjust(arr:np.ndarray[np.float64], val:float, loc:LocSpec, outer:LocSpec=None)->np.ndarray[np.float64]:
    """Adjust arr by val at loc across outer"""
    # convert loc and outer into boolean masks
    if val == 0.0:
        return arr.copy()
    loc = _loc_reshape(loc, arr.shape)
    outer = _loc_reshape(outer, arr.shape)
    outer *= (loc.sum(axis=1) != 0)[:,np.newaxis]
    # get the inverse of loc within outer
    inv = outer ^ loc
    if np.any(loc & inv):
        raise ValueError("loc is not subset of outer")
    # allocate arrays so row wise sums can be done on masks
    arrl = np.zeros(arr.shape, dtype=np.float64)
    arri = np.zeros(arr.shape, dtype=np.float64)
    arrl[loc], arri[inv] = arr[loc], arr[inv]
    suml, sumi = arrl.sum(axis=1)[:,np.newaxis], arri.sum(axis=1)[:,np.newaxis] # row wise sums
    # remove rows that have total of zero, so change by 0
    nonzeros = (suml != 0.0) & (sumi != 0.0)
    loc *= nonzeros
    inv *= nonzeros
    suml, sumi = suml[nonzeros], sumi[nonzeros]
    # compute amount to adjust
    tot = suml + sumi
    val = np.exp(np.log(suml)*(1-val)/val)
    # adjust arrays
    arr[loc] = (arrl/suml*val*tot)[loc] # rescale function
    arr[inv] = (arri/sumi*(1-val)*tot)[inv]
    return arr


def trans_adjust(model:hm.h2mm_model, val:float, loc:LocSpec, 
               outer:None|tuple[tuple[int,int],...]|np.ndarray[np.bool_]=None)->hm.h2mm_model:
    r"""
    Return a model with trans array "adjusted" by factor val at locations loc,
    if outer specified, only touch outer


    Parameters
    ----------
    model : hm.h2mm_model
        Model to adjust.
    val : float
        factor by which to shift specified locatons in trans array.
    loc : tuple[tuple[int|tuple[int,...],...],...]|np.ndarray[np.bool\_]
        locations to adjust, either boolean mask, or tuple of trans index tuples.
        See :attr:`LocSpec` for general format description.
    outer : None|tuple[tuple[int,int],...]|np.ndarray[np.bool\_], optional
        Locations to allow change in values in trans array. 
        See :attr:`LocSpec` for general format description.
        The default is None.

    Returns
    -------
    hm.h2mm_model
        trans adjusted |H2MM| model.

    """
    prior, trans, obs = model.prior.copy(), model.trans.copy(), model.obs.copy()
    trans = _arr_adjust(trans, val, loc, outer)
    return hm.h2mm_model(prior, trans, obs)


def obs_adjust(model:hm.h2mm_model, val:float, loc:LocSpec, outer:LocSpec=None)->np.ndarray[np.float64]:
    r"""
    Return a model with obs array "adjusted" by factor val at locations loc,
    if outer specified, only touch outer


    Parameters
    ----------
    model : hm.h2mm_model
        Model to adjust.
    val : float
        factor by which to shift specified locatons in obs array.
    loc : tuple[tuple[int,int],...]|np.ndarray[np.bool\_]
        locations to adjust, either boolean mask, or tuple of trans index tuples.
        See :attr:`LocSpec` for general format description
    outer : None|tuple[tuple[int,int],...]|np.ndarray[np.bool\_], optional
        Locations to allow change in values in obs array. 
        See :attr:`LocSpec` for general format description
        The default is None.

    Returns
    -------
    hm.h2mm_model
        obs adjusted |H2MM| model.

    """
    prior, trans, obs = model.prior.copy(), model.trans.copy(), model.obs.copy()
    obs = _arr_adjust(obs, val, loc, outer)
    return hm.h2mm_model(prior, trans, obs)


_adjust_funcs = {'prior':prior_adjust, 'trans':trans_adjust, 'obs':obs_adjust}


def _adjust_func(adjust:AFunc, model:hm.h2mm_model, adj_kwargs:dict, eval_kwargs:dict, 
                 targ:float, indexes:DArray, times:TArray, x:np.ndarray)->float:
    """Function for scipy.optimize.fminbound evaluation of ll error"""
    out = adjust(model, x, **adj_kwargs).evaluate(indexes, times, inplace=False, **eval_kwargs)
    return (out.loglik - targ)**2


def outer_trans_mask(loc:tuple[int,int], shape:tuple[int,int])->np.ndarray[np.bool_]:
    r"""
    Create "outer" mask for adjusting trans array by single specified location and shape.
    Used to generate the outer argument for trans_adjust so that transition rates
    do not change the diagonal.

    Parameters
    ----------
    loc : tuple[int,int]
        Transition rate to vary in adjust function.
    shape : tuple[int,int]
        Shape of trans array, should be (n, n), unput as tuple so can use trans.shape.

    Returns
    -------
    out : np.ndarray[np.bool\_]
        Mask of points not to change.

    """
    shape = (shape, shape) if isinstance(shape, Number) else shape
    loc = _loc_reshape(loc, shape)
    out = np.zeros(shape, dtype=np.bool_)
    I = np.eye(*shape, dtype=np.bool_)
    for i in range(shape[0]):
        if np.any(loc[i,:] & I[i,:]):
            out[i,:] = True 
        else:
            out[i,:] = loc[i,:] | I[i,:]
    return out
    

def _out_ll_models(locs:tuple[tuple[int,...],...], models:Sequence[hm.h2mm_model], shape:tuple[int,...])->np.ndarray[hm.h2mm_model]:
    """Take sequence of models from ll calcualation and coerce them into an appropriately shaped array"""
    out = np.empty(shape, dtype=np.object_)
    for loc, model in zip(locs, models):
        out[loc] = model
    return out


def evalutate_ll_error(model:hm.h2mm_model, indexes:DArray, times:TArray, 
                       adjust:AFunc|Literal['prior','trans','obs'], 
                       targ:float=0.5, eval_kwargs:dict=None, bound_kwargs:dict=None,
                       **kwargs)->tuple[hm.h2mm_model,hm.h2mm_model]:
    """
    Compute the error of value(s) in a model based on data. Finds the points
    where the loglikelihood is targ (default 0.5) less than the model.

    Parameters
    ----------
    model : hm.h2mm_model
        |H2MM| model (``hm.h2mm_model``) to compute the model error based on the
        loglikelihood error.
    indexes : DArray
        Indexes of photons in data, handed to ``hm.h2mm_model.evaluate`` .
    times : TArray
        Times of photons in data, handed to ``hm.h2mm_model.evaluate``.
    adjust : Literal['prior','trans','adjust']|AFunc
        Which part ('trans', 'obs', or 'prior') of model to compute error.
        May also be callable to provide customized adjustments.
        Signature must be ``adjust(model:float, val:float, **kwargs)`` and return
        a model. the ``val`` argument is a float that is varied by
        ``scipy.optimize.fminbound`` and ``kwargs`` are the input to ``adj_kwargs``.
    targ : float, optional
        Target decrease in loglikelihood for error computation. The default is 0.5.
    eval_kwargs : dict, optional
        Kwargs handed to ``hm.h2mm_model.evaluate``. The default is None.
    bound_kwargs : dict, optional
        Kwargs  handed to 
        |scipyoptimize|. 
        The default is None.
    **kwargs : Any, optional
        Kwargs handed to model adjust function.
    

    Returns
    -------
    low : hm.h2mm_model
        Model with adjustment in the negative direction with loglik targ less
        than model.
    high : hm.h2mm_model
        Model with adjustment in the positive direction with loglik targ less
        than model.

    """
    adjust = _adjust_funcs[adjust] if isinstance(adjust, str) else adjust
    # render None kwargs to correct mutable types
    bound_kwargs = dict() if bound_kwargs is None else bound_kwargs
    eval_kwargs = dict() if eval_kwargs is None else eval_kwargs
    # compute target value, ie ideal loglik
    targ = model.evaluate(indexes, times, inplace=False, **eval_kwargs).loglik - targ
    # create partial function to use in optimzation
    func = partial(_adjust_func, adjust, model, kwargs, eval_kwargs, targ, indexes, times)
    low = optimize.fminbound(func, 0.0, 0.5, **bound_kwargs)
    high = optimize.fminbound(func, 0.5, 1.0, **bound_kwargs)
    if isinstance(low, Number):
        low = adjust(model, low, **kwargs).evaluate(indexes, times, inplace=False, **eval_kwargs)
    else:
        low = (adjust(model, low[0], **kwargs).evaluate(indexes, times, inplace=False, **eval_kwargs),) + low[1:]
    if isinstance(high, Number):
        high = adjust(model, high, **kwargs).evaluate(indexes, times, inplace=False, **eval_kwargs)
    else:
        high = (adjust(model, high[0], **kwargs).evaluate(indexes, times, inplace=False, **eval_kwargs),) + high[1:]
    return low, high


def statepath_ll_error(data:PhotonDataS, statepath:Param, 
                       adjust:AFunc|Literal['prior','trans','obs']='trans', 
                       loc:LocSpec=None, adj_kwargs:dict=None, targ:float=0.5,
                       eval_kwargs:dict=None, bound_kwargs:dict=None
                       )->tuple[hm.h2mm_model|np.ndarray[hm.h2mm_model],hm.h2mm_model|np.ndarray[hm.h2mm_model]]:
    r"""
    Evaluate the estimated error of a model in a :class:`StatePathBase` based 
    ``Param`` by point when given location decreases in loglikelihood by 
    a target value (targ, usually 0.5).

    Parameters
    ----------
    data : PhotonDataS
        Source data object.
    statepath : Param
        ``StatePathBase`` based ``Param`` defining a |H2MM| model and burst
        selection for which to evaluate error.
    adjust : AFunc|Literal['prior','trans','obs'], optional
        Which array ('prior', 'trans', 'obs') to adjust, or an "adjust" function,
        which takes the signature 
        ``adjust(model:hm.h2mm_model, val:float, **kwargs)``. 
        The default is 'trans'.
    loc : tuple[int,int], optional
        Location to adjust, generally tuple specifying location in array. 
        This argument is ignored if ``adjust`` is not one of 'prior', 'trans' or 'obs'
        The default is None.
    adj_kwargs : dict|Sequence[dict], optional
        Kwargs handed to adjustment function. The default is None.
    targ : float, optional
        Target decrease in loglikelihood for error computation. The default is 0.5.
    eval_kwargs : dict, optional
        Kwargs handed to ``hm.h2mm_model.evaluate``. The default is None.
    bound_kwargs : dict, optional
        Kwargs  handed to |scipyoptimize|.
        The default is None.

    Raises
    ------
    ValueError
        Bad input value.

    Returns
    -------
    low : hm.h2mm_model | np.ndarray[hm.h2mm_model]
        Lower error based on loglikelihood computation.
    high : hm.h2mm_model  | np.ndarray[hm.h2mm_model]
        Higher error based on loglikelihood computation.

    """
    if issubclass(statepath.tp, Dwells):
        statepath = statepath.parents['statepath']
    model = statepath.params['model']
    indexc, timec = Column(statepath, 'indexpath'), Column(statepath, 'timepath')
    if hasattr(data, 'concatenate_column'):
        indexes = data.concatenate_column(indexc)
        times = data.concatenate_column(timec)
    else:
        indexes = data.get_column(indexc)
        times = data.get_column(timec)
    if isinstance(adjust, str):
        adjust = _adjust_funcs.get(adjust, None)
        if adjust is None:
            raise ValueError("Invalid str adjust function")
    adj_kwargs = dict() if adj_kwargs is None else adj_kwargs
    if loc is None:
        adj_kwargs = repeat(adj_kwargs) if isinstance(adj_kwargs, dict) else adj_kwargs
        if adjust == trans_adjust:
            locs = tuple(product(range(model.nstate), range(model.nstate)))
            shape = (model.nstate, model.nstate)
            lak = zip(locs, (_dict_update({'outer':outer_trans_mask(l, shape)}, adk) 
                             for l, adk in zip(locs, adj_kwargs)))
        elif adjust == obs_adjust:
            locs = tuple(product(range(model.nstate), range(model.ndet)))
            shape = (model.nstate, model.ndet)
            lak = zip(locs, adj_kwargs)
        elif adjust == prior_adjust:
            locs = tuple(range(model.nstate))
            shape = (model.nstate, )
            lak = zip(locs, adj_kwargs)
        else:
            raise ValueError("Invalid str adjust function")
        low, high = zip(*(evalutate_ll_error(model, indexes, times, adjust, 
                                             loc=l, targ=targ, 
                                             eval_kwargs=eval_kwargs, 
                                             bound_kwargs=bound_kwargs, **ak) 
                          for l, ak in lak))
        low, high = _out_ll_models(locs, low, shape), _out_ll_models(locs, high, shape)
    else:
        low, high = evalutate_ll_error(model, indexes, times, adjust, loc=loc,
                                       targ=targ, eval_kwargs=eval_kwargs, 
                                       bound_kwargs=bound_kwargs, **adj_kwargs)
    return low, high
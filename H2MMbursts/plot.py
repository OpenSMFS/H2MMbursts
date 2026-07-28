#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Paul David Harris
# Purpose: plotting wrappers for H2MM functions
# Created: 12/12/2025
# Modified: 21/04/2025
"""
Plot
====

Functions for plotting various |H2MM| based parameters.

Some functions serve as overlays to overlay values derived from models with those
calculated from dwells or bursts.

Others are for plotting statistical discriminators etc.

.. |H2MM| replace:: H\ :sup:`2`\ MM
.. |StatePath| replace:: :class:`H2MMbursts.modeltables.StatePath`
.. |Dwells| replace:: :class:`H2MMbursts.modeltables.Dwells`
.. |Axes| replace:: `plt.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`__
.. |Line2D| replace:: `Line2D <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html>`__
.. |LineCollection| replace:: `mpl.collections.LineCollection <https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection>`__
.. |plttext| replace:: `plt.Text <https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text>`__
.. |PathCollection| replace:: `mpl.collections.PathCollection <https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.PathCollection>`__
.. |ListedColorMap| replace:: `mpl.colors.ListedColorMap <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.ListedColormap.html>`__
.. |axvline| replace:: `ax.axvline <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvline.html>`__
.. |axhline| replace:: `ax.axvline <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axhline.html>`__
.. |axtitle| replace:: `ax.set_title <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html>`__
.. |axxlabel| replace:: `ax.set_xlabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html>`__
.. |axylabel| replace:: `ax.set_ylabel <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html>`__
.. |axscatter| replace:: `ax.scatter <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html>`__
.. |axannotate| replace:: `ax.annotate <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html>`__
.. |Annotation| replace:: `plt.Annotation <https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Annotation>`__
.. |axlegend| replace:: `ax.legend <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html>`__
"""
from typing import Any, Literal
from collections.abc import Sequence, Callable
from itertools import permutations, product, combinations
from numbers import Real

import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.transforms import Transform, Affine2DBase
from matplotlib.text import Annotation

from smfbursts.datamodel.utils import _dict_update
from smfbursts.datamodel.tables import Param, Table, Column
from smfbursts.datamodel.plot import _rescale_value, _check_ax
from smfbursts.photondata import PhotonData, PhotonDataS
from smfbursts.plot import _get_factor, axaltspan

from .modeltables import (
    StatePathBase, Dwells, 
    calc_BIC, calc_BICph, calc_BICp, calc_ICL, calc_ICLph, calc_pathBIC, calc_pathBICph
    )


def _as_array_like(val, shape:tuple[int,int], dtype:np.dtype, name:str)->np.ndarray:
    """Coerce val into a numpy array of shape shape, name used for error message"""
    v = np.asarray(val, dtype=dtype)
    if v.ndim == 0:
        out = np.empty(shape, dtype=dtype)
        if dtype == np.object_:
            for ij in product(*(range(s) for s in shape)):
                out[ij] = val
        else:
            out[tuple(slice(None) for _ in shape)] = val
    elif v.shape != shape:
        raise ValueError(f"{name} cannot be converted to array of shape {shape}")
    else:
        out = v
    return out


def hist_model(col:Column, statepath:StatePathBase|Param=None, data:PhotonDataS=None,
               ax:plt.Axes=None, strict:bool=True, rescale:Real=1.0, 
               orientation:Literal['horizontal','vertical']='vertical',
               line_kwargs:Sequence[dict]=None,
               **kwargs)->tuple[mpl.lines.Line2D,...]:
    """
    Overlay vertical or horizontal lines of the value defined by col of the states
    in model defined by statepath.
    
    This is useful for placing markers in a histogram of col indicating the
    position of states in the model.

    Parameters
    ----------
    col : Column
        Column defines parameter being plotted on relevant axis.
    statepath : StatePathBase | Param, optional
        |StatePath| based param which contains the |H2MM| model from which to 
        calculate values. The default is None.
    data : PhotonDataS, optional
        If necessary, to provide additional scaling etc. information, the 
        source of data used to optimize the model/for which a histogram is
        also being plotting of col in data on the same axis. The default is None.
    ax : plt.Axes, optional
        The |Axes| into which the lines will be plotted. 
        If None, use current axes.
        The default is None.
    strict : bool, optional
        If :code:`True`, raise an error when streams in col are not compatible 
        with the given statepath's streams selection. The default is True.
    rescale : Real, optional
        Rescale value of data axis, same system as rescale in other hist type plots. 
        The default is 1.0.
    orientation : Literal['horizontal','vertical'], optional
        Whether lines should be oriented vertically or horizontally.
        Options are 'horizontal' or 'vertical'
        The default is 'vertical'.
    line_kwargs : Sequence[dict], optional
        Keyword arguments passed per state to |axvline| or |axhline|. 
        The default is None.
    **kwargs : Any
        Keword arguments handed to |axvline| or |axhline|.

    Raises
    ------
    TypeError
        Cannot determine model from inputs.

    Returns
    -------
    tuple[mpl.lines.Line2D,...]
        tuple of |Line2D| objects 1 for each state.

    """
    if statepath is None:
        if issubclass(col.param.tp, StatePathBase):
            statepath = col.param
        elif issubclass(col.base_param.tp, Dwells):
            statepath = col.base_param.parents['statepath']
        else:
            raise TypeError("must specify statepath if col is not derived from StatePath param (cannot identify h2mm_model)")
    ax = _check_ax(ax)
    vals = statepath.model_value(col, origin=data, strict=strict)
    mx = _rescale_value(vals, rescale)
    func = ax.axvline if orientation == 'vertical' else ax.axhline
    line_kwargs = dict() if line_kwargs is None else line_kwargs
    line_kwargs = _as_array_like(line_kwargs, mx.shape, np.object_, 'line_kwargs')
    return tuple(func(x, **_dict_update(kwargs, line_kwargs[i])) for i, x in enumerate(mx))


def scatter_model(colx:Column, coly:Column, statepath:StatePathBase|Param=None, 
                  data:PhotonDataS=None, ax:plt.Axes=None, strict:bool=True, 
                  rescale:tuple[Real,Real]=None, **kwargs:Any)->mpl.collections.PathCollection:
    """
    

    Parameters
    ----------
    colx : Column
        Column to plot model values in x-axis.
    coly : Column
        Column to plot model values in y-axis.
    statepath : StatePathBase| Param, optional
        |StatePath| based ``Param`` the model of whose state values to plot.
        If ``colx`` or ``coly`` are themselves |StatePath| tables or |StatePath| 
        based ``Param`` s, this may be ommitted and the model infered from 
        ``colx`` or ``coly`` The default is None.
    data : PhotonDataS, optional
        Data for which models are being plotted/assesed against, may be used
        to infer additional values. The default is None.
    ax : plt.Axes, optional
        DESCRIPTION. The default is None.
    strict : bool, optional
        If :code:`True`, raise an error when streams in col are not compatible 
        with the given statepath's streams selection. The default is True.
    rescale : tuple[Real,Real], optional
        Rescale values used in scatter of colx vs coly. The default is None.
    **kwargs : Any
        Additional kwargs passed to |axscatter|.

    Raises
    ------
    TypeError
        Cannot determine model.

    Returns
    -------
    mpl.collections.PathCollection
        |PathCollection| resulting from |axscatter| of model values.

    """
    if statepath is None:
        for col in (colx, coly):
            if issubclass(col.param.tp, StatePathBase):
                statepath = col.param
                break
            elif issubclass(col.base_param.tp, Dwells):
                statepath = col.base_param.parents['statepath']
                break
    if statepath is None:
        raise TypeError("must specify statepath if neither colx nor coly is not derived from StatePath param (cannot identify h2mm_model)")
    if isinstance(statepath, Param):
        if data is None:
            raise TypeError("Must supply data if statepath is infered from columns or specified as Param")
        statepath = data.get_table(statepath)
        if isinstance(statepath, tuple):
            statepath = statepath[0]
    rescale = 1.0 if rescale is None else rescale
    rescalex, rescaley = rescale if isinstance(rescale, tuple) else (rescale, rescale)
    ax = _check_ax(ax)
    mx, my = statepath.model_values(colx, coly, origin=data, strict=strict)
    mx, my = _rescale_value(mx, rescalex), _rescale_value(my, rescaley)
    return ax.scatter(mx, my, **kwargs)


class ConnectAffine(Affine2DBase):
    """
    Special blended-like transform for connecting two points. X axis becomes
    line between the two points and Y axis is perpenticular, with unit of ytrans.
    
    Parameters
    ----------
    source : (float, float)
        x, y position of source (x=0 in new coordinate position)
    dest : (float, float)
        x, y position of destination (x=1 in new coordinate position)
    point_trans : matplotlib.transforms.Transform
        Transform of source and dest points
    ytrans : matplotlib.transforms.Transform
        Transform of y coordinate, must be linear, only uses y direction
    retract : float, optional
        Amount, in units of y transform to "retract" from line between source and dest
        on which to move source and dest points. The default is 0.0.
    """
    def __init__(self, source:tuple[float,float], dest:tuple[float,float], 
                 point_trans:Transform, ytrans:Transform, 
                 retract:float=0.0, **kwargs):
        super().__init__(**kwargs)
        self._s = source
        self._d = dest
        self._rt = retract
        self._point_trans = point_trans
        self.set_children(point_trans)
        self._ytrans = ytrans
        self.set_children(ytrans)
        self._inverted = None
        self._invalid = 0

    def get_matrix(self):
        source = self._point_trans.transform(self._s)
        dest = self._point_trans.transform(self._d)
        delta = dest - source
        mag = np.linalg.norm(delta)
        yscale = (self._ytrans.transform((0,1)) - self._ytrans.transform((0,0)))[1] / mag
        self._mtx = np.empty((3,3), dtype=np.float64)
        self._mtx[2,2] = 1.0
        self._mtx[:2,1] = yscale*delta[::-1]
        self._mtx[0,1] *= -1
        self._mtx[:2,0] = delta*(1-2*yscale*self._rt)
        self._mtx[:2,2] = source + delta * yscale * self._rt
        self._inverted = None
        self._invalid = 0
        return self._mtx


def _get_dpi(ax:plt.Axes)->Transform:
    """Get figure dpi transform from Axis"""
    return ax.figure.dpi_scale_trans


def _get_transData(ax:plt.Axes)->Transform:
    """Get transData transform from Axis"""
    return ax.transData


def _get_transAxes(ax:plt.Axes)->Transform:
    """Get transAxes transform from Axis"""
    return ax.transAxes


def _get_xaxis(ax:plt.Axes)->Transform:
    """Get blended xaxis transform from Axis"""
    return ax.get_xaxis_transform()


def _get_yaxis(ax:plt.Axes)->Transform:
    """Get blended xaxis transform from Axis"""
    return ax.get_yaxis_transform()


_get_transform_dict = {'dpi':_get_dpi, 'data':_get_transData, 'axis':_get_transAxes, 
                       'xaxis':_get_xaxis, 'yaxis':_get_yaxis}


def _get_transform(ax:plt.Axes, transform:str|Transform)->Transform:
    """Retrive specified tranform type from axis"""
    return _get_transform_dict[transform](ax) if isinstance(transform, str) else transform


def _get_lpos(source:np.ndarray[np.float64], dest:np.ndarray[np.float64], frac:float=0.5)->np.ndarray[np.float64]:
    source, dest = np.asarray(source), np.asarray(dest)
    return source + frac * (dest - source)


_normalize_rename = {'ha':'horizontalalignment', 'va':'verticalalignment'}


def _normalize_annotate(dct:dict)->dict:
    """
    Ensure consistent naming of ha/horizontalalignment and va/verticalallignment
    always chooses the latter
    """
    if dct is None:
        return dict()
    dct = dct.copy()
    for name, rename in _normalize_rename.items():
        if name in dct:
            dct[rename] = dct.pop(name)
    return dct


def _update_kws(kws:dict, *args:dict)->dict:
    """Update kwargs dict with any number of dictionaries, each normalizing allignment nomenclature"""
    out = _normalize_annotate(kws)
    for arg in args:
        out.update(_normalize_annotate(arg))
    return out


def _get_distances(pos:np.ndarray[np.float64])->np.ndarray[np.float64]:
    """Compute distances between paris of points in pos"""
    d = np.empty(pos.shape[:-1]+pos.shape[:-1], dtype=np.float64)
    for i, j in product(range(pos.shape[0]), range(pos.shape[1])):
        d[i,j,:,:] = np.sqrt(np.sum((pos - pos[i,j])**2, axis=2))
    return d


TransformSpec = Literal[tuple(_get_transform_dict.keys())]|Transform

def annotate_transition(source:tuple[float,float], dest:tuple[float,float], label:str, 
                        ax:plt.Axes=None, retract:float=0.1, transform:TransformSpec='data',
                        offset:float=0.0, offset_transform:TransformSpec='dpi', 
                        label_pos:float=0.5, rotate_text:bool=True, source_style:str='-', dest_style:str='-|>', 
                        source_kwargs:dict=None, dest_kwargs:dict=None, arrowprops:dict=None,
                        **kwargs)->tuple[Annotation,Annotation]:
    r"""
    Plot transition rate arrows between specified source and dets (x, y) points.
    
    This function is used by 
    :func:`hist_model_trans_arrows` and :func:`scatter_model_trans_arrows` to
    plot transition rates between states.

    Parameters
    ----------
    source : tuple[float,float]
        x,y location of "source" of transition based, in unit specified by transform
    dest : tuple[float,float]
        x,y location of "destination" of transition based, in unit specified by transform.
    label : str
        Label to give transition (typically a transition rate, converted to string).
    ax : plt.Axes, optional
        |Axes| in which to place transition arrows. The default is None.
    retract : float, optional
        Amount, in transform units by which to retract start/end of arrows. The default is 0.1.
    transform : TransformSpec, optional
        Transform to use to define points of source and dest. 
        If a string, get the transform from the axis, options are:
            
            - ``"dpi"`` use figure dpi coordinates (``ax.figure.dpi_scale_trans``)
            - ``"data"`` use the axis data coordinates (``ax.transData``)
            - ``"axis"`` use the axis data coordinates (``ax.transAxes``)
            - ``"xaxis"`` use the xdata blended transform (``ax.get_xaxis_transform()``)
            - ``"yaxis"`` use the ydata blended transform (``ax.get_yaxis_transform()``)
        
        The default is 'data'.
    offset : float, optional
        Ammount by which to offset the arrow, in units of offset_transform. The default is 0.0.
    offset_transform : TransformSpec, optional
        Transform used to offset the transition arrow, same options as for ``transform``. 
        The default is 'dpi'.
    label_pos : float, optional
        Fraction of distance from source to dest to place the label text. The default is 0.5.
    rotate_text : bool, optional
        Whether or not to rotate the text to match the angle of the arrows. 
        The default is True.
    source_style : str, optional
        arrowstyle of source transition arrow (from source to label). The default is '-'.
    dest_style : str, optional
        arrowstyle of dest transition arrow (from label to dest). The default is '-\|>'.
    source_kwargs : dict, optional
        Keyword arguments passed to |axannotate| for creating source arrow.
        The default is None.
    dest_kwargs : dict, optional
        Keyword arguments passed to |axannotate| for creating dest arrow.
        The default is None.
    **kwargs : Any
        Additional kwargs passed to both source and dest arrows of |axannotate|.

    Returns
    -------
    source_arrow : Annotation
        |Annotation| object of arrow from source to label.
    dest_arrow : Annotation
        |Annotation| object of arrow from label to dest.

    """
    arrowprops = dict() if arrowprops is None else arrowprops
    source, dest = np.asarray(source), np.asarray(dest)
    ax = _check_ax(ax)
    transform = _get_transform(ax, transform)
    offset_transform = _get_transform(ax, offset_transform)
    if source[0] <= dest[0]:
        ct = ConnectAffine(source, dest, transform, offset_transform, retract=retract)
        sp, lp, dp = (0.0, offset), (label_pos, offset), (1.0, offset)
    else:
        ct = ConnectAffine(dest, source, transform, offset_transform, retract=retract)
        sp, lp, dp = (1.0, -offset), (1-label_pos, -offset), (0.0, -offset)
    kws = {'transform_rotates_text':rotate_text, 'horizontalalignment':'center', 
           'verticalalignment':'center', 'rotation_mode':'anchor'}
    source_kwargs = _update_kws(kws, {'arrowprops':_dict_update({'arrowstyle':source_style}, 
                                                                arrowprops)}, 
                                kwargs, source_kwargs)
    dest_kwargs = _update_kws(kws, {'arrowprops':_dict_update({'arrowstyle':dest_style}, 
                                                              arrowprops)}, 
                              kwargs, dest_kwargs)
    source_arrow = ax.annotate(label, sp, xytext=lp, xycoords=ct, textcoords=ct, **source_kwargs)
    dest_arrow = ax.annotate(label, dp, xytext=lp, xycoords=ct, textcoords=ct, **dest_kwargs)
    return source_arrow, dest_arrow


def hist_model_trans_arrows(col:Column, statepath:Param|StatePathBase=None, 
                            data:PhotonDataS=None, ax:plt.Axes=None, 
                            strict:bool=True, rescale:Real=None, 
                            positions:np.ndarray[np.float64]=None, orientation:str='vertical', 
                            min_rate:float=0.0, rotate_text:bool=True, label_fmt:str|np.ndarray[np.str_]='3.0f', 
                            label_pos:np.ndarray[np.float64]|float=0.5, 
                            transform:TransformSpec=None, offset:float=0.1, 
                            offset_transform:TransformSpec='dpi',
                            source_style:str='-', dest_style:str='-|>',
                            source_kwargs:Sequence[Sequence[dict]]|dict=None, 
                            dest_kwargs:Sequence[Sequence[dict]]|dict=None, 
                            match_pos:bool=True, retract:float=None,
                            **kwargs)->np.ndarray[tuple[mpl.text.Annotation,mpl.text.Annotation]]:
    r"""
    Plot arrows with transition rates between each state in statepath, using 
    values of model for col on a histogram-like plot.

    Parameters
    ----------
    col : Column
        Column of the data axis (x/y depends on value of ``orientation``, defaut
        will plot along x-axis), if column is derived from StatePath-like Param,
        can infer model.
    statepath : Param | StatePathBase, optional
        Either StatePath or Param[StatePath] defining the model. 
        If Param[StatePath], must specify data keyword argument to define the
        data upon which the param is defined. The default is None.
    data : PhotonData | PhotonDataList, optional
        The data source, necessary for determining clock-rate of transition matrix. 
        Required if statepath is supplied as :class:`Param` or infered from column.
        The default is None.
    ax : plt.Axes, optional
        |Axes| in which to plot arrows. The default is None.
    strict : bool, optional
        If :code:`True`, raise an error when streams in col are not compatible 
        with the given statepath's streams selection. The default is True.
    rescale : Real, optional
        Rescale values used in hist of col. The default is None.
    positions : np.ndarray[np.float64], optional
        Position of each transition along non-data axis. The default is None.
    orientation : str, optional
        Orientation plot, if ``'vertical'`` plot col along x-axis (as for a vertically
        oriented histogram), if ``'horizontal'``, plot col along y-axis. 
        The default is 'vertical'.
    min_rate : float, optional
        Slowest transition rate to plot, if rate is less than min_rate, 
        then omit transition. The default is 0.0.
    rotate_text : bool, optional
        Whether the labels should be rotated according to the angle of 
        the connecting arrows. The default is True.
    label_fmt : str, optional
        Format strings (e.g 3.1f or e) for transition rates. The default is '3.0f'.
    label_pos : np.ndarray[np.float64] | float, optional
        Fractional position of position of label along axis between locations 
        of states. The default is 0.5.
    transform : Literal['dpi','data','axis','xaxis','yaxis'] | Transform, optional
        Transform to use as for placing points. This argument is passed on to
        :func:`annotate_transition`.
        The default is None.
    offset : float, optional
        Distance to offset arrows perpendicular to the line between states. 
        The default is 0.1.
    offset_transform : Literal['dpi','data','axis','xaxis','yaxis'] | Transform, optional, optional
        Transform to use as dimention of offset. 
        This argument is passed on to :func:`annotate_transition`
        The default is 'dpi'.
    source_style : str, optional
        arrowstyle argument of the "source" arrow (arrow from source state to label). 
        The default is '-'.
    dest_style : str, optional
        arrowstyle argument of the "dest" arrow (arrow from label to destination state). 
        The default is '-\|>'.
    source_kwargs : Sequence[Sequence[dict]] | dict, optional
        Dictionary or transition-wise array of dictionaries of keyword arguments
        for `ax.annotate <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html>`_
        for the arrow pointing from the "source" state to the label
        The default is None.
    dest_kwargs : Sequence[Sequence[dict]] | dict, optional
        Dictionary or transition-wise array of dictionaries of keyword arguments
        for `ax.annotate <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html>`_
        for the arrow pointing from the the label state to the "destination" state.
        The default is None.
    match_pos : bool, optional
        Whether to align transition pairs 
        (only if label_pos is specified with single number). 
        The default is True.
    retract : float, optional
        Amount by which to offset the starting/ending points of arrows from states. 
        If :code:`None` (default) will use value of offset
        The default is None.
    **kwargs : Any
        Keyword arguments fed to all calls to 
        `ax.annotate <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html>`_ .

    Raises
    ------
    TypeError
        Invalid combination of arguments, namely no data argument supplied when
        statepath supplied as :class:`Param` instead of :class:`Table`.

    Returns
    -------
    out : np.ndarray[tuple[mpl.text.Annotation,mpl.text.Annotation]]
        Array of |Annotation| objects produced by each transition.

    """
    if statepath is None:
        if issubclass(col.param.tp, StatePathBase):
            statepath = col.param
        elif issubclass(col.base_param.tp, Dwells):
            statepath = col.base_param.parents['statepath']
        else:
            raise TypeError("must specify statepath if col is not derived from StatePath param (cannot identify h2mm_model)")
    if statepath is None:
        raise TypeError("must specify statepath if col is not derived from StatePath param (cannot identify h2mm_model)")
    # get statepath table **if specified as statepath and data**
    if isinstance(statepath, Param):
        if data is None:
            raise TypeError("must specify data when statepath is given as Param")
        statepath = data.get_table(statepath)
        statepath = statepath if isinstance(statepath, Table) else statepath[0]
    ax = _check_ax(ax)
    model = statepath.transrate
    mx = _rescale_value(statepath.model_value(col, origin=data, strict=strict), 1.0 if rescale is None else rescale)
    source_kwargs = dict() if source_kwargs is None else source_kwargs
    dest_kwargs = dict() if dest_kwargs is None else dest_kwargs
    source_kwargs = dict() if source_kwargs is None else source_kwargs
    dest_kwargs = dict() if dest_kwargs is None else dest_kwargs
    retract = offset if retract is None else retract
    source_style = _as_array_like(source_style, model.shape, np.object_, 'source_style')
    dest_style = _as_array_like(dest_style, model.shape, np.object_, 'dest_style')
    source_kwargs = _as_array_like(source_kwargs, model.shape, np.object_, 'source_kwargs')
    dest_kwargs = _as_array_like(dest_kwargs, model.shape, np.object_, 'dest_kwargs')
    label_fmt = _as_array_like(label_fmt, model.shape, np.object_, 'label_fmt')
    label_pos = _as_array_like(label_pos, model.shape, np.float64, 'label_pos')
    if transform is None:
        transform = 'xaxis' if orientation == 'vertical' else 'yaxis'
    transform = _as_array_like(transform, model.shape, np.object_, 'transform')
    offset = _as_array_like(offset, model.shape, np.float64, 'offset')
    offset_transform = _as_array_like(offset_transform, model.shape, np.object_, 'offset_transform')
    retract = _as_array_like(retract, model.shape, np.float64, 'retract')
    min_rate = _as_array_like(min_rate, model.shape, np.float64, 'min_rate')
    rotate_text = _as_array_like(rotate_text, model.shape, np.bool_, 'rotate_text')
    comb = tuple((i, j) for i, j in combinations(range(model.shape[0]), 2) if model[i,j] >= min_rate[i,j] or model[j,i] >= min_rate[i,j])
    pos = np.linspace(0.0, 1.0, len(comb)+2)[1:-1]
    if positions is None:
        positions = -np.ones(model.shape, dtype=np.float64)
        for (i, j), p in zip(comb, pos):
            positions[i,j] = p
            positions[j,i] = p
    else:
        positions = np.asarray(positions, dtype=np.float64).reshape(model.shape)
    out = np.empty(model.shape, dtype=np.object_)
    for i, j in permutations(range(model.shape[0]), 2):
        if model[i,j] < min_rate[i,j] or label_pos[i,j] == -1.0:
            continue
        off = 0.0 if model[j,i] < min_rate[j,i] or label_pos[j,i] == -1.0 else offset[i,j]
        ps = mx[i], positions[i,j]
        pd = mx[j], positions[j,i]
        if orientation == 'horizontal':
            ps, pd = ps[::-1], pd[::-1]
        out[i,j] = annotate_transition(ps, pd, ('%%%s' % label_fmt[i,j]) % model[i,j],
                                       ax=ax, retract=retract[i,j], transform=transform[i,j], 
                                       offset=off, offset_transform=offset_transform[i,j],
                                       label_pos=label_pos[i,j] if match_pos and i < j else 1 - label_pos[i,j],
                                       rotate_text=rotate_text[i,j],
                                       source_style=source_style[i,j], dest_style=dest_style[i,j],
                                       source_kwargs=source_kwargs[i,j], dest_kwargs=dest_kwargs[i,j], **kwargs)
    return out


def scatter_model_trans_arrows(colx:Column, coly:Column, 
                               statepath:Param|StatePathBase=None, data:PhotonDataS=None, 
                               ax:plt.Axes=None, strict:bool=True,
                               rescale:Real|tuple[Real,Real]=None,
                               min_rate:float=0.0, rotate_text=True, label_fmt:str='3.0f', 
                               label_pos:np.ndarray[np.float64]|float=0.5, transform:TransformSpec='data',
                               offset:float=0.1, offset_transform:TransformSpec='dpi',
                               source_style:str='-', dest_style='-|>',
                               source_kwargs:Sequence[Sequence[dict]]|dict=None, 
                               dest_kwargs:Sequence[Sequence[dict]]|dict=None, 
                               check_locs:bool=True, match_pos:bool=True, retract:float=None, 
                               **kwargs)->np.ndarray[tuple[mpl.text.Annotation,mpl.text.Annotation]]:
    r"""
    Plot arrows with transition rates between each state in statepath, using 
    values of model for colx and coly on a scatter plot.

    Parameters
    ----------
    colx : Column
        Column of x-axis.
    coly : Column
        column of y-axis.
    statepath : Param | StatePathBase, optional
        Either StatePath or Param[StatePath] defining the model. 
        If Param[StatePath], must specify data keyword argument to define the
        data upon which the param is defined. The default is None.
    data : PhotonDataS, optional
        Required if statepath is supplied as :class:`Param`, the data source,
        necessary for determining clock-rate of transition matrix. The default is None.
    ax : plt.Axes, optional
        |Axes| in which to plot arrows. The default is None.
    strict : bool, optional
        If :code:`True`, raise an error when streams in col are not compatible 
        with the given statepath's streams selection. The default is True.
    rescale : Real | tuple[Real,Real], optional
        Rescale values used in scatter of colx vs coly. If specify a single value
        use same rescale value for both axes. The default is None.
    min_rate : float | np.ndarray[np.float64], optional
        Slowest transition rate to plot, if rate is less than min_rate, 
        then omit transition. The default is 0.0.
    rotate_text : bool | np.ndarray[np.bool\_], optional
        Whether the labels should be rotated according to the angle of 
        the connecting arrows. The default is True.
    label_fmt : str | np.ndarray[np.str\_], optional
        DESCRIPTION. The default is '3.0f'.
    label_pos : float | np.ndarray[np.float64], optional
        Fractional position of position of label along axis between locations 
        of states. The default is 0.5.
    transform : TransformSpec | np.ndarray[TransformSpec], optional
        Transform to use as for placing points. If specified as an array, use 2D
        array of [source, dest] specs. Internally each transition is passed to
        :func:`annotate_transition`.
        The default is 'data'.
    offset : float | np.ndarray[np.float64], optional
        Distance to offset arrows perpendicular to the line between states. 
        The default is 0.1.
    offset_transform : TransformSpec | np.ndarray[TransformSpec], optional
        Transform to use as dimention of offset. 
        Internally passed to :func:`annotate_transition`. Same syntax as ``transform``.
        The default is 'dpi'.
    source_style : str, optional
        arrowstyle argument of the "source" arrow (arrow from source state to label). 
        The default is '-'.
    dest_style : TYPE, optional
        arrowstyle argument of the "dest" arrow (arrow from label to destination state). 
        The default is '-\|>'.
    source_kwargs : Sequence[Sequence[dict]] | dict, optional
        Dictionary or transition-wise array of dictionaries of keyword arguments
        for `ax.annotate <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html>`_
        for the arrow pointing from the "source" state to the label
        The default is None.
    dest_kwargs : Sequence[Sequence[dict]] | dict, optional
        Dictionary or transition-wise array of dictionaries of keyword arguments
        for `ax.annotate <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html>`_
        for the arrow pointing from the the label state to the "destination" state.
        The default is None.
    check_locs : bool, optional
        Whether to check that the locations of each label are likely to overlap. 
        The default is True.
    match_pos : bool, optional
        Whether to align transition pairs 
        (only if label_pos is specified with single number). 
        The default is True.
    retract : float, optional
        Amount by which to offset the starting/ending points of arrows from states. 
        If :code:`None` (default) will use value of offset
        The default is None.
    **kwargs : Any
        Keyword arguments fed to all calls to 
        `ax.annotate <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html>`_ .

    Raises
    ------
    TypeError
        Invalid combination of arguments, namely no data argument supplied when
        statepath supplied as :class:`Param` instead of :class:`Table`.

    Returns
    -------
    out : np.ndarray[tuple[mpl.text.Annotation,mpl.text.Annotation]]
        Array of |Annotation| objects produced by each transition.

    """
    if statepath is None:
        for col in (colx, coly):
            if issubclass(col.param.tp, StatePathBase):
                statepath = col.param
                break
            elif issubclass(col.base_param.tp, Dwells):
                statepath = col.base_param.parents['statepath']
                break
    if statepath is None:
        raise TypeError("must specify statepath if neither colx nor coly is not derived from StatePath param (cannot identify h2mm_model)")
    # get statepath table **if specified as statepath and data**
    if isinstance(statepath, Param):
        if data is None:
            raise TypeError("must specify data when statepath is given as Param")
        statepath = statepath = data.get_table(statepath)
        statepath = statepath if isinstance(statepath, Table) else statepath[0]
    model = statepath.transrate
    rescale = 1.0 if rescale is None else rescale
    rescalex, rescaley = rescale if isinstance(rescale, tuple) else (rescale, rescale)
    mx, my = statepath.model_values(colx, coly, origin=data, strict=strict)
    mx, my = _rescale_value(mx, rescalex), _rescale_value(my, rescaley)
    # setup array for source and dest kwargs
    source_kwargs = dict() if source_kwargs is None else source_kwargs
    dest_kwargs = dict() if dest_kwargs is None else dest_kwargs
    source_kwargs = dict() if source_kwargs is None else source_kwargs
    dest_kwargs = dict() if dest_kwargs is None else dest_kwargs
    retract = offset if retract is None else retract
    source_style = _as_array_like(source_style, model.shape, np.object_, 'source_style')
    dest_style = _as_array_like(dest_style, model.shape, np.object_, 'dest_style')
    source_kwargs = _as_array_like(source_kwargs, model.shape, np.object_, 'source_kwargs')
    dest_kwargs = _as_array_like(dest_kwargs, model.shape, np.object_, 'dest_kwargs')
    label_fmt = _as_array_like(label_fmt, model.shape, np.object_, 'label_fmt')
    label_pos = _as_array_like(label_pos, model.shape, np.float64, 'label_pos')
    transform = _as_array_like(transform, model.shape, np.object_, 'transform')
    offset = _as_array_like(offset, model.shape, np.float64, 'offset')
    offset_transform = _as_array_like(offset_transform, model.shape, np.object_, 'offset_transform')
    retract = _as_array_like(retract, model.shape, np.float64, 'retract')
    min_rate = _as_array_like(min_rate, model.shape, np.float64, 'min_rate')
    rotate_text = _as_array_like(rotate_text, model.shape, np.bool_, 'rotate_text')
    ax = _check_ax(ax)
    # compute fractional label positions, 2 cases: single (or none) label pos specified, or both forward and reverse
    if check_locs:
        # first get expected locations of each label if 
        pos = np.empty(model.shape+(2,), dtype=np.float64)
        for i, j in product(range(model.shape[0]), range(model.shape[1])):
            pos[i,j,:] = _get_lpos((mx[i], my[i]), (mx[j], my[j]), label_pos[i,j])
        # compute distances
        dist = _get_distances(pos)
        for (i, j), (k, l) in combinations(combinations(range(model.shape[0]), 2), 2):
            if (model[i,j] < min_rate[i,j] or label_pos[i,j] == -1.0) and (model[j,i] < min_rate[j,i] or label_pos[j,i] == -1.0):
                continue
            if dist[i,j,k,l] < offset[i,j] or dist[i,j,k,l] < offset[j,i]:
                label_pos[i,j] = 0.2
                label_pos[j,i] = 0.8
    else:
        if isinstance(label_pos, Real):
            label_pos = np.ones(model.shape, dtype=np.float64)*label_pos
    out = np.empty(model.shape, dtype=np.object_)
    for i, j in permutations(range(model.shape[0]), 2):
        if model[i,j] < min_rate[i,j] or label_pos[i,j] == -1.0:
            continue
        off = 0.0 if model[j,i] < min_rate[j,i] or label_pos[j,i] == -1.0 else offset[i,j]
        out[i,j] = annotate_transition((mx[i], my[i]), (mx[j], my[j]), ('%%%s' % label_fmt[i,j]) % model[i,j],
                                       ax=ax, retract=retract[i,j], transform=transform[i,j], 
                                       offset=off, offset_transform=offset_transform[i,j],
                                       label_pos=label_pos[i,j] if match_pos and i < j else 1 - label_pos[i,j],
                                       rotate_text=rotate_text[i,j],
                                       source_style=source_style[i,j], dest_style=dest_style[i,j],
                                       source_kwargs=source_kwargs[i,j], dest_kwargs=dest_kwargs[i,j], **kwargs)
    return out


def _bstate_names(n:int, names:Sequence[str], max_comb:int)->str:
    """Gate name of burst bcode (which states present by viterbi) from bitcode n"""
    nstate = np.bitwise_count(n)
    if nstate <= max_comb:
        return ' and '.join(name for i, name in enumerate(names) if n & (1<<i))
    return f'{nstate} states'


def ordered_comb(order:Sequence[int], depth:int)->tuple[int,...]:
    """
    Reimplementation of combinations, order ensured to iterate last dimenstion
    fastest.
    """
    for i, o in enumerate(order):
        if depth != 1:
            yield from ((o, ) + c for c in ordered_comb(order[i+1:], depth-1))
        else:
            yield (o, )


def make_bstate_cmap(names:Sequence[str], colors:Sequence[str|tuple[float,float,float]], 
                     max_comb:int=1, order:Sequence[int]=None)->tuple[ListedColormap,ListedColormap,list[str]]:
    r"""
    Function for creating |ListedColorMap|\ s and list of state names for plotting
    ``smf.plot.scatter`` where points are colored by either the burst state combination
    or dwell state.
    
    The brief summary\:
    
    - ``names`` should be names of states in order found in |H2MM| model, 
    - ``colors`` according to order of display, 
      ie [1st state 2nd state, ... combinations of states, ...]
    - ``max_comb`` specifies when naming each combination of states stops
    - ``order`` specifies desired reordering or states for display, e.g. if the
      second state in |H2MM| order should be at the "top" of the list it ``order``
      shoudl be [1, ...] (remember indexing from 0, so second state is index 1).
    
    This function is intended to be used in conjunction with :func:`state_labels_dwells`.
    
    The requried inputs are a list of names (``names``) for each state 
    (number of states infered by length of this ``names``), 
    a sequence of colors (``colors``), used to assign a color to each state and
    combination of states. (minimum length descibed later in function description)
    
    The output is 2 |ListedColorMap|\ s, the first for use with plotting dwells
    colored by dwell state, the other for plotting bursts colored by the combination
    of states (regardless of order) present within bursts, and the final argument
    is a list of names for each possible combinations of states.
    
    The ``max_comb`` keyword argument determines how states are named, and how
    many distinct colors are in the colormaps. Initially combinations of states 
    are named as a combination of the names within e.g. "State A and StateB",
    but when the number of states exceeds ``max_comb``, then regardless of which
    states are present the combination is named "N states". 
    The order of in the second colormap and list of names is the same, and
    is ordered according to the sorting of the "bstates" column bitcodes, therefore
    the order goes: 
    
        ``[state0, state1, state1and2, state3, state1and3, state1and2and3,...]``
    
    If N is the number of states in ``names`` then the first N colors in colors
    are assigned to each state, the order is set by the ``order`` keyword argument
    The size of colors is determined by number of states in ``names`` and ``max_comb``.
    
    Below in an example of how to setup this function
    
    .. code-block::
        
        state_order = [3,1,2,0]
        state_names = ['Acceptor only', 'Closed', 'Donor Only', 'Open']
        clr = ['r', 'b', 'purple', 'pink', 'g', 'y', 'c', 'm', 'aqua', 'grey', 'gold', 'salmon']
        dcmap, bcmap, nlist = bhm.plot.make_bstate_cmap(state_names, clr, max_comb=2, order=state_order)

    
    if we examine nlist we will have
    
    >>> nlist
    ['',
     'Acceptor only',
     'Closed',
     'Acceptor only and Closed',
     'Donor Only',
     'Acceptor only and Donor Only',
     'Closed and Donor Only',
     '3 states',
     'Open',
     'Acceptor only and Open',
     'Closed and Open',
     '3 states',
     'Donor Only and Open',
     '3 states',
     '3 states',
     '4 states']
    
    Note that the 0th element is an empty string, because the ordering is
    from 0 to number of unique state combinations, but 0b0 means no bursts, so
    an empty dwell, which makes no sense, but is necessary based on how
    ``smf.plot.colorcategory`` assumes numbers start at 0.
    
    Then to plot dwells we would execute
    
    .. code-block::
        
        fig, ax = plt.subplots()
        smf.plot.scatter(data, dwellE, dwellS, point_func=smf.plot.colorcategory, point_cols=dwellstate, point_kwargs={'cmap':dcmap}, ax=ax)
        handles = bhm.plot.state_labels_dstate(state_names, dcmap, order=state_order, ax=ax)
        ax.legend(handles=handles)
    
    for plotting bursts, and add a legend we would execute
    
    .. code-block::
        
        fig, ax = plt.subplots()
        smf.plot.scatter(data, burstE, burstS, point_func=smf.plot.colorcategory, point_cols=burstbstate, point_kwargs={'cmap':bcmap}, ax=ax)
        handles = bhm.plot.state_labels_bursts(nlist, bcmap, order=state_order, ax=ax)
        ax.legend(handles=handles)
        
    
    Parameters
    ----------
    names : Sequence[str]
        Names for each state in model, should be same length as number of states.
    colors : Sequence[str|tuple[float,float,float]]
        Sequence of valid matplotlib color definitions, 1 for each state/combination
        of states expected.
    max_comb : int, optional
        Maximum number of states to give unique names, if a given bcode has
        more than max_comb states, the name will be "# states". The default is 1.
    order : Sequence[int], optional
        Order of states for display. 
        (Since order of states is fixed in |H2MM| model, must reorder with order)
        The default is None.

    Returns
    -------
    dcmap : ListedColormap
        |ListedColorMap| for use in cmap of ``smf.plot.scatter`` when 
        ``point_cols`` column is "state" of |Dwells|.
    bcmap : ListedColormap
        |ListedColorMap| for use in cmap of ``smf.plot.scatter`` when 
        ``point_cols`` column is "bstates" of |StatePath|.
    nlist : list[str]
        List of names for each bstate code, includes dummy 0 at index 0.

    """
    nlist = [_bstate_names(i, names, max_comb) for i in range(1<<len(names))]
    order = range(len(names)) if order is None else order
    clist, i = ["#000000" for _ in range(1<<len(names))], 0
    for nc in range(1,len(names)+1):
        if nc <= max_comb:
            for cmb in ordered_comb(order, nc):
                clist[sum(1<<c for c in cmb)] = colors[i]
                i += 1
        else:
            for cmb in combinations(range(len(names)), nc):
                clist[sum(1<<c for c in cmb)] = colors[i]
            i += 1
    bcmap = ListedColormap(clist)
    if max_comb > 0:
        dcmap = ListedColormap([colors[i] for i in order]) 
    else:
        ListedColormap([colors[0] for _ in range(len(names))])
    return dcmap, bcmap, nlist


def _order_state(x:int, order:Sequence[int])->tuple[int, int]:
    """
    Create tuple of (number of state, state order) for ordering a bitcode
    x according first to the number of states, and then the hierarchy 
    establised by order
    """
    sz = np.bitwise_count(x)
    od = sum(((x&(1<<o))>>o)<<i for i, o in enumerate(order))
    return sz, od


def state_labels_dwells(nlist:Sequence[str], dcmap:ListedColormap, ax:None=None,
                        order:Sequence[int]=None, **kwargs)->list[mpl.collections.PatchCollection]:
    r"""
    Use this function to create a list that can be used as the handels argument
    to |axlegend| for a scatter plot where the colors were set by the dwell
    "state" column.
    
    
    This function creates a set of "dummy" scatter plots 
    (which all have single point at [nan, nan]) with a label assigne by nlist.
    
    This function is meant to be used in conjuction with the 
    :func:`make_bstate_cmap` function and ``smf.plot.scatter`` 
    where the ``point_func`` kwarg is ``smf.plotcolorcategory`` the ``point_cols``
    kwarg is the "state" column of |Dwells|.
    
    Minimal code example below\:
    
    .. code-block::
        
        dcmap, bcmap, nlist = bhm.plot.make_bstate_cmap(state_names, clr, order=state_order)
        smf.plot.scatter(data, colx, coly, point_func=smf.plot.colorcategory, point_cols=bstate, ax=ax, point_kwargs={'cmap':dcmap})
        leg = bhm.plot.state_labels(nlist, dcmap, ax=ax, order=state_order)
        ax.legend(handles=leg)
    
    
    See :func:`make_bstate_cmap` for description of how ``state_names``, ``clr``
    and ``order`` should be used.

    Parameters
    ----------
    nlist : Sequence[str]
        List of names for each state/state combination, order is defined by order
        of bstate (ie 0b1, 0b10...) This is usually created by :func:`make_bstate_cmap`.
    dcmap : ListedColormap
        A |ListedColorMap| that should be of same nubmer of states as nlist defining color
        for each single bit true bstate (ie single state), is first output of
        :func:`make_bstate_cmap`
    ax : plt.Axes, optional
        |Axes| in which to create invisible labels, if None, use current axis. 
        The default is None.
    order : Sequence[int], optional
        Order in which to add states, based on order of states, not bcode. 
        The default is None.
    **kwargs : Any
        Additional kwargs passed to |axscatter|.

    Returns
    -------
    list[mpl.collections.PatchCollection]
        The |PathCollection|\ s of each dummy point for creating a legend, handed
        to handles argument of |axlegend|.

    """
    ax = _check_ax(ax)
    xy = np.array([np.nan])
    order = np.r_[0:len(dcmap.colors)]
    return [ax.scatter(xy, xy, color=dcmap(i), label=nlist[1<<i], **kwargs) for i in order]


def state_labels_bursts(nlist:Sequence[str], bcmap:ListedColormap, ax:plt.Axes=None, 
                        order:Sequence[int]=None, **kwargs:Any)->list[mpl.collections.PatchCollection]:
    r"""
    Use this function to create a list that can be used as the handels argument
    to |axlegend| for a scatter plot where the colors were set by the burst
    "bstates" column.
    
    
    This function creates a set of "dummy" scatter plots 
    (which all have single point at [nan, nan]) with a label assigne by nlist.
    
    This function is meant to be used in conjuction with the 
    :func:`make_bstate_cmap` function and ``smf.plot.scatter`` 
    where the ``point_func`` kwarg is ``smf.plotcolorcategory`` the ``point_cols``
    kwarg is the "bstates" column of |StatePath|.
    
    Minimal code example below\:
    
    .. code-block::
        
        dcmap, bcmap, nlist = bhm.plot.make_bstate_cmap(state_names, clr, order=state_order)
        smf.plot.scatter(data, colx, coly, point_func=smf.plot.colorcategory, point_cols=bstate, ax=ax, point_kwargs={'cmap':bcmap})
        leg = bhm.plot.state_labels(nlist, bcmap, ax=ax, order=state_order)
        ax.legend(handles=leg)
    
    
    See :func:`make_bstate_cmap` for description of how ``state_names``, ``clr``
    and ``order`` should be used.

    Parameters
    ----------
    nlist : Sequence[str]
        List of names for each state/state combination, order is defined by order
        of bstate (ie 0b1, 0b10...) This is usually created by :func:`make_bstate_cmap`.
    bcmap : ListedColormap
        A |ListedColorMap| that should be of same length as nlist defining color
        for each bstate column bitcode possibility.
    ax : plt.Axes, optional
        |Axes| in which to create invisible labels, if None, use current axis. 
        The default is None.
    order : Sequence[int], optional
        Order in which to add states, based on order of states, not bcode. 
        The default is None.
    **kwargs : Any
        Additional kwargs passed to |axscatter|.

    Returns
    -------
    list[mpl.collections.PatchCollection]
        The |PathCollection|\ s of each dummy point for creating a legend, handed
        to handles argument of |axlegend|\ .

    """
    ax = _check_ax(ax)
    labels = set(nlist)
    locs = [nlist.index(l) for l in labels if l]
    order = list(range(len(nlist))) if order is None else order
    reorder = sorted(locs, key=lambda x: _order_state(x, order))
    xy = np.array([np.nan])
    return [ax.scatter(xy, xy, color=bcmap(i), label=nlist[i], **kwargs) for i in reorder]


def burst_index(data:PhotonData, param:Param, burst:int, ax:plt.Axes=None, 
                time_direction:Literal['x','y']='x', 
                rescale:Real=None, zerostart:bool=False, 
                index_pos:dict[tuple[int,...],float]=None, 
                index_kwargs:dict[tuple[int,...],dict[str,Any]]=None, label_kwargs:dict[str,Any]=None,
                alt_span:bool=False, alt_span_kwargs:dict=None,
                **kwargs:Any)->list[mpl.collections.PathCollection,...,plt.Text,list[list[mpl.patches.Rectangle]]]:
    """
    Similar to the ``smf.plot.burst_dets`` function, plots photons in a single
    burst, with position defined by the |H2MM| indexes defined by the 
    |StatePath| based ``Param``.

    Parameters
    ----------
    data : PhotonData
        Source data for plot.
    param : Param
        Definition of time range, must be |StatePath| based ``Param``.
    burst : int
        Burst number.
    ax : plt.Axes, optional
        |Axes| in which to plot the burst photons, if None, use current axis. 
        The default is None.
    time_direction : str, optional
        Direction of time axis. The default is 'x'.
    rescale : Real, optional
        Rescale factor for time axis. The default is None.
    zerostart : bool, optional
        If :code`True`, photon times start at 0. The default is False.
    index_pos : dict[tuple[int,...],float], optional
        Dictionary of index to position mappings. The default is None.
    index_kwargs : dict[tuple[int,...],dict[str,Any]], optional
        Dictionary of kwargs dictionaries passed per detector key to |axscatter|. 
        The default is None.
    label_kwargs : dict[str,Any], optional
        Keyword arguments passed to |axxlabel| or |axylabel|. 
        The default is None.
    alt_span : bool, optional
        Whether to plot the alternation periods with :func:`axaltspan`.
        Should only be :code:`True` if ``data`` is usALEX data.  
        The default is False.
    alt_span_kwargs : dict, optional
        Keyword arguments handed to :func:`axaltspan`.
    **kwargs : Any
        Universal kwargs hannded to |axscatter| for each plot of index class.

    Returns
    -------
    list[mpl.collections.PathCollection,...,plt.Text,list[list[mpl.patches.Rectangle]]]
        |PathCollection| of each plotted index, ending with the |plttext| object from
        |axxlabel| or |axylabel|.

    """
    ax = _check_ax(ax)
    label_kwargs = dict() if label_kwargs is None else label_kwargs
    if index_pos is None:
        if index_kwargs is None:
            index_pos = {i:i for i in range(param.params['model'].ndet)}
        else:
            index_pos = {k:i for i, k in enumerate(index_kwargs.keys())}
    if index_kwargs is None:
        index_kwargs = {k:dict() for k in index_pos.keys()}
    for i, (times, dets) in enumerate(zip(data.iter_column(Column(param, 'timepath')),
                                          data.iter_column(Column(param, 'indexpath')))):
        if i == burst:
            break        
    out = list()
    unit, factor = _get_factor(data, rescale)
    time_direction = time_direction.lower()
    if alt_span:
        alt_span_kwargs = dict() if alt_span_kwargs is None else alt_span_kwargs
        spans = axaltspan(data, times[0], times[-1], ax=ax, 
                          time_shift= times[0] if zerostart else 0, rescale=rescale,
                          time_direction=time_direction, **alt_span_kwargs)
    times = times - times[0] if zerostart else times
    for k, p in index_pos.items():
        mask = np.isin(dets, k)
        x, y = times[mask]*factor, np.repeat(p, mask.sum())
        xy = x, y
        if time_direction in ('y', 'vertical'):
            xy = xy[::-1]
        out.append(ax.scatter(*xy, **_dict_update(index_kwargs.get(k, dict()), kwargs)))
    if time_direction in ('y', 'vertical'):
        out.append(ax.set_ylabel(unit, **label_kwargs))
    else:
        out.append(ax.set_xlabel(unit, **label_kwargs))
    if alt_span:
        out.append(spans)
    return out


def burst_state(data:PhotonData, param:Param, burst:int, ax:plt.Axes=None, 
                time_direction:Literal['x','y']='x', 
                rescale:Real=None, zerostart:bool=False, 
                state_pos:dict[tuple[int,...],float]=None, 
                state_kwargs:dict[tuple[int,...],dict[str,Any]]=None, label_kwargs:dict[str,Any]=None,
                alt_span:bool=False, alt_span_kwargs:dict=None,
                **kwargs:Any)->list[mpl.collections.PathCollection,...,plt.Text]:
    """
    Similar to the ``smf.plot.burst_dets`` function, plots photons in a single
    burst, with position defined by the state defined by the 
    |StatePath| based ``Param``.

    Parameters
    ----------
    data : PhotonData
        Source data for plot.
    param : Param
        Definition of time range, must be |StatePath| based ``Param``.
    burst : int
        Burst number.
    ax : plt.Axes, optional
        |Axes| in which to plot the burst photons, if None, use current axis. 
        The default is None.
    time_direction : str, optional
        Direction of time axis. The default is 'x'.
    rescale : Real, optional
        Rescale factor for time axis. The default is None.
    zerostart : bool, optional
        If :code`True`, photon times start at 0. The default is False.
    state_pos : dict[tuple[int,...],float], optional
        Dictionary of state index to position mappings. The default is None.
    state_kwargs : dict[tuple[int,...],dict[str,Any]], optional
        Dictionary of kwargs dictionaries passed per state index to |axscatter|. 
        The default is None.
    label_kwargs : dict[str,Any], optional
        Keyword arguments passed to |axxlabel| or |axylabel|. 
        The default is None.
    alt_span : bool, optional
        Whether to plot the alternation periods with :func:`axaltspan`.
        Should only be :code:`True` if ``data`` is usALEX data.  
        The default is False.
    alt_span_kwargs : dict, optional
        Keyword arguments handed to :func:`axaltspan`.
    **kwargs : Any
        Universal kwargs hannded to |axscatter| for each plot of index class.

    Returns
    -------
    list[mpl.collections.PathCollection,...,plt.Text]
        |PathCollection| of each plotted index, ending with the |plttext| object from
        |axxlabel| or |axylabel|.

    """
    ax = _check_ax(ax)
    label_kwargs = dict() if label_kwargs is None else label_kwargs
    if state_pos is None:
        if state_kwargs is None:
            state_pos = {i:i for i in range(param.params['model'].nstate)}
        else:
            state_pos = {k:i for i, k in enumerate(state_kwargs.keys())}
    if state_kwargs is None:
        state_kwargs = {k:dict() for k in state_pos.keys()}
    for i, (times, dets) in enumerate(zip(data.iter_column(Column(param, 'timepath')),
                                          data.iter_column(Column(param, 'statepath')))):
        if i == burst:
            break        
    out = list()
    unit, factor = _get_factor(data, rescale)
    time_direction = time_direction.lower()
    if alt_span:
        alt_span_kwargs = dict() if alt_span_kwargs is None else alt_span_kwargs
        spans = axaltspan(data, times[0], times[-1], ax=ax, 
                          time_shift= times[0] if zerostart else 0, rescale=rescale,
                          time_direction=time_direction, **alt_span_kwargs)
    times = times - times[0] if zerostart else times
    for k, p in state_pos.items():
        mask = np.isin(dets, k)
        x, y = times[mask]*factor, np.repeat(p, mask.sum())
        xy = x, y
        if time_direction in ('y', 'vertical'):
            xy = xy[::-1]
        out.append(ax.scatter(*xy, **_dict_update(state_kwargs.get(k, dict()), kwargs)))
    if time_direction in ('y', 'vertical'):
        out.append(ax.set_ylabel(unit, **label_kwargs))
    else:
        out.append(ax.set_xlabel(unit, **label_kwargs))
    if alt_span:
        out.append(spans)
    return out


def _lc_kwarg(state0:int, state1:int, name:str, val:dict[int|tuple[int,int]:Any], kwargs:dict[str:Any])->Any:
    if (state0, state1) in val:
        return val[(state0, state1)]
    if state0 == state1 and state1 in val:
        return val[state0]
    if name[:-1] in kwargs:
        return kwargs[name[:-1]]
    return plt.rcParams.get(f'line.{name[:-1]}', None)


def _lc_kwargs(states:np.ndarray[np.int8], name:str, val:dict[int|tuple[int,int]:Any], 
               kwargs:dict[str:Any])->Sequence[Any]:
    return [_lc_kwarg(state0, state1, name, val, kwargs) 
            for state0, state1 in zip(states[:-1], states[1:])]


def burst_statepath(data:PhotonData, statepath:Param, burst:int, ax:plt.Axes=None,
                    time_direction:Literal['x','y']='x', rescale:Real=None, zerostart:bool=False,
                    state_pos:np.ndarray[np.float64]=None, 
                    state_kwargs:dict[str:dict[int|tuple[int,int]:Any]]=None,
                    tlabel:str=None, tlabel_kwargs:dict[str,Any]=None,
                    slabel:str=None, slabel_kwargs:dict[str,Any]=None, 
                    **kwargs:Any)->tuple[LineCollection,plt.Text,plt.Text]:
    """
    Plot the *Viterbi* statepath of a single burst as connected lines 
    (showing ranges of dwells). Works well in conjuction with :func:`burst_index`
    to display the *Viterbi* path of a burst and its cooresponding photons.

    Parameters
    ----------
    data : PhotonData
        Source of data for bursts.
    statepath : Param
        Definition of burst search and |H2MM| model for *Viterbi* path.
    burst : int
        Burst to plot.
    ax : plt.Axes, optional
        |Axes| in which to plot *Viterbi* path. The default is None.
    time_direction : Literal['x','y'], optional
        Direction in which time progresses, either 'x' or 'y'.
        The default is 'x'.
    rescale : Real, optional
        Rescale factor for time axis. The default is None.
    zerostart : bool, optional
        If :code:`True`, times are relative to fisrt photon in burst. 
        The default is False.
    state_pos : np.ndarray[np.float64], optional
        Array of positions for each state, if None, use integer spacing. 
        The default is None.
    state_kwargs : dict[str:dict[int|tuple[int,int]:Any]], optional
        Dictionary of keyword arguments, each a value dictionary of state indexes
        (int), or tuple of transitions (2-tuple of int, int)
        to |LineCollection|. Dictionary is used to unwrap based on state path
        the appropriate sequence for the given kwarg (outer dictionary) based
        on keys available. If a key is not available for the necessary element
        of a sequence, the value defaults to None.
        The default is None.
    tlabel : str, optional
        Name of time axis label, if label is not to be set, should be :code:`False`. 
        The default is None.
    tlabel_kwargs : dict[str,Any], optional
        Kwargs handed to time label function (|axxlabel| or |axylabel|).
        The default is None.
    slabel : str, optional
        Name of state axis label, if label is not to be set, should be :code:`False`. 
        The default is None.
    slabel_kwargs : dict[str,Any], optional
        Kwargs handed to state axis label function (|axxlabel| or |axylabel|).
        The default is None.
    **kwargs : Any
        Additional kwargs all handed directly to |LineCollection|.

    Raises
    ------
    ValueError
        Wrong size of ``state_pos`` argument.

    Returns
    -------
    lc : mpl.collections.LineCollection
        |LineCollection| of the statepath.
    tlbl : plt.Text
        |plttext| object of time axis label.
    slbl : plt.Text
        |plttext| object of state axis label.

    """
    ax = _check_ax(ax)
    tlabel_kwargs = dict() if tlabel_kwargs is None else tlabel_kwargs
    slabel_kwargs = dict() if slabel_kwargs is None else slabel_kwargs
    if issubclass(statepath.tp, Dwells):
        statepath = statepath.parents['statepath']
    unit, factor = _get_factor(data, rescale)
    tlabel = unit if tlabel is None else tlabel
    slabel = 'state' if slabel is None else slabel
    state_pos = np.arange(statepath.params['model'].nstate) if state_pos is None else np.atleast_1d(state_pos).reshape(-1)
    if (err:=statepath.params['model'].nstate) != state_pos.size:
        raise ValueError(f"state_pos must be of same size as number of states in model, got {state_pos.size} and {err}")
    for i, (times, states) in enumerate(zip(data.iter_column(Column(statepath, 'timepath')),
                                            data.iter_column(Column(statepath, 'statepath')))):
        if i == burst:
            break
    times = times - times[0] if zerostart else times
    times = times*factor
    statesp = state_pos[states]
    # orient time direction
    if time_direction.lower() in ('x', 'horizontal'):
        x, y = times, statesp
        tlbl = ax.set_xlabel(tlabel, **tlabel_kwargs) if tlabel is not False else None
        slbl = ax.set_ylabel(slabel, **slabel_kwargs) if slabel is not False else None
    else:
        x, y = statesp, times
        tlbl = ax.set_ylabel(tlabel, **tlabel_kwargs) if tlabel is not False else None
        slbl = ax.set_xlabel(slabel, **slabel_kwargs) if slabel is not False else None
    # plot either as linecollection or as ax.plot
    if state_kwargs is not None:
        segments = np.array([np.array([x[:-1], y[:-1]]), np.array([x[1:],y[1:]])]).transpose(2,0,1)
        if isinstance(state_kwargs, dict):
            kws = {k:_lc_kwargs(states, k, v, kwargs) for k, v in state_kwargs.items()}
        for k, v in kwargs.items():
            if f'{k}s' in kws:
                continue
            kws.setdefault(k, v)
        if 'transform' not in kws:
            kws['transform'] = ax.transData
        lc = LineCollection(segments, **kws)
        ax.add_collection(lc)
    else:
        lc = ax.plot(x, y, **kwargs)
    return lc, tlbl, slbl


_calc_stat_discs = dict(BIC=calc_BIC, BICph=calc_BICph, BICp=calc_BICp, 
                        ICL=calc_ICL, ICLph=calc_ICLph, 
                        pathBIC=calc_pathBIC, pathBICph=calc_pathBICph)
_stat_disc_thresh = {calc_BIC:0.005, calc_BICph:0.005, calc_BICp:0.005, 
                     calc_ICL:0.0, calc_ICLph:0.0,
                     calc_pathBIC:0.0, calc_pathBICph:0.0}
_stat_disc_titles = {calc_BIC:"$BIC$", calc_BICph:r"$BIC\:ph^{-1}$", calc_BICp:r"$BIC^{\prime}$", 
                     calc_ICL:"$ICL$", calc_ICLph:r"$ICL\:ph^{-1}$",
                     calc_pathBIC:"$BIC_{path}$", calc_pathBICph:r"$BIC_{path}\:ph^{-1}$"}

#: Generic alis for type hinting for statistical discriminator options
StatDisc = Literal[tuple(_calc_stat_discs.keys())]|Callable[[Sequence[Param]],np.ndarray[np.float64]]


def _sort_stat_disc(stat:StatDisc)->Callable[[Sequence[Param]],np.ndarray[np.float64]]:
    """Get function for computing statistical discriminator from string or function, used to process stat input"""
    if isinstance(stat, str):
        if stat not in _calc_stat_discs:
            raise ValueError(f"Invalid statistical discriminator string {stat}")
        stat = _calc_stat_discs[stat]
    if stat is not None and not callable(stat):
        raise ValueError("Statistical discriminator must be callable")
    return stat


def scatter_model_statdisc(data:PhotonDataS, statepaths:Sequence[Param], ax:plt.Axes=None, 
                           stat:StatDisc='ICL', highlight:StatDisc|Literal['same',False,None]='same', 
                           thresh:float|Literal[None,True,'auto']='auto', 
                           highlight_kwargs:dict[str:Any]=None, 
                           title:None|bool|str=None, title_kwargs:dict[str:Any]=None,
                           xlabel:Literal[None,False]|str='States', xlabel_kwargs:dict[str:Any]=None,
                           **kwargs:Any
                           )->tuple[mpl.collections.PatchCollection,mpl.collections.PatchCollection,plt.Text,plt.Text]:
    r"""
    Plot any statistical discriminator for a sequence of |StatePath| based ``Param`` s.
    
    This is useful to assess the ideal model from :meth:`H2MMbursts.StatePath.optimize_models`
    
    .. code-block::
        
        statepaths = bhm.StatePath.optimize_models(data, to_state=4, max_state=8)
        bhm.plot.scatter_model_statdisc(data, statepaths, stat='ICL')

    Parameters
    ----------
    data : PhotonDataS
        Data on which statepaths are based 
        (use this as source of data to compute the given statistical discriminator).
    statepaths : Sequence[Param]
        Sequence of |StatePath| based ``Param`` defining the models for which the
        statistical discriminator should be computed and plotted, generally each
        should be of a different number of states but otherwise the same parameters.
    ax : plt.Axes, optional
        |Axes| in which to place plot. If None, use current axes.
        The default is None.
    stat : StatDisc, optional
        Which statistical discriminator to use may be one of :
            
            - ``"BIC"`` the Bayes Information Criterion
            - ``"BICph"`` the Bayes Information Criterion divided by the number of photons
            - ``"BICp"`` the modified Bayes Information Criterion (:math:`BIC^{\prime}`)
              from `Lerner 2018 <https://doi.org/10.1063/1.5004606>`_
            - ``"ICL"`` the integrated complete likelihood
            - ``"ICLph"`` the integrated complete likelihood divided by the number of photons
            - ``"pathBIC"`` the Bayes Information Criterion of the most likely state-path
            - ``"pathBICph"`` the Bayes Information Criterion of the most likely state-path
              divided by the number of photons
        
        Or a callable with the signature ``calc_disc(origin:PhotonDataS, statepaths:Sequence[Param])``
        Which should return the statistical discriminator values as a 1D array
        of size statepaths, given the data in origin.
        The default is 'ICL'.
    highlight : StatDisc | {'same', False, None}, optional
        The statistical discriminator to use to specify which point to highlight
        as ideal, with the threshold specified in thresh. 
        If ``'same'``, then use same statistical discriminator as ``stat``, if ``False``
        then do not highlight, otherwise use same format as argument of ``stat``.
        The default is 'same'.
    thresh : float | {None, True, 'auto'}, optional
        Threshold for statustical discriminator, if 'auto' infer from which 
        discriminator used. The default is 'auto'.
    highlight_kwargs : dict[str:Any], optional
        Kwargs handed to |axscatter| for plotting the highlighted (ideal) number of states. 
        The default is None.
    title : None|bool|str, optional
        Title to give to plot, if None or True, automatically set title based on 
        statistical discriminator, if False, do not set title, if str, use string
        directly as title. The default is None.
    title_kwargs : dict[str:Any], optional
        Keyword arguments handed to |axtitle|. The default is None.
    xlabel : Literal[None,False]|str, optional
        xlabel of plot, if :code:`None` or :code:`False` do not plot. The default is 'States'.
    xlabel_kwargs : dict[str:Any], optional
        Kwargs handed to |axxlabel|.
        The default is None.
    **kwargs : Any
        Additional kwargs are handed to |axscatter| for plotting the statistical
        discriminators for each statepath.

    Returns
    -------
    scat : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of non-highlighted statistical discriminators.
    hl : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of highlighted statistical discriminator.
    ttl : plt.Text | None
        |plttext| object of title.
    xttl : plt.Text | None
        |plttext| object of x-axis label.

    """
    states = np.array([sp.params['model'].nstate for sp in statepaths])
    stat = _sort_stat_disc(stat)
    vals = stat(data, statepaths)
    ax = _check_ax(ax)
    scat, hl, ttl, xttl = None, None, None, None
    if highlight is not None and highlight is not False:
        highlight = stat if isinstance(highlight, str) and highlight == 'same' else _sort_stat_disc(highlight)
        hlv = vals if highlight is stat else highlight(data, statepaths)
        thresh = _stat_disc_thresh.get(highlight, 0.0) if thresh in (True, 'auto') else thresh
        hloc = np.array([np.argmin(hlv)]) if thresh is None else np.argwhere((hlv-np.min(hlv)) <= thresh).reshape(-1)
        if hloc.size and np.any(states[hloc] != np.max(states)):
            mask = states == np.min(states[hloc])
            imask = ~mask
            highlight_kwargs = dict() if highlight_kwargs is None else highlight_kwargs
            scat = ax.scatter(states[imask], vals[imask], **kwargs)
            hl = ax.scatter(states[mask], vals[mask], **highlight_kwargs)
        else:
            scat = ax.scatter(states, vals, **kwargs)
    else:
        scat = ax.scatter(states, vals, **kwargs)
    title = _stat_disc_titles.get(stat, stat.__name__) if title is None or title is True else title
    title_kwargs = dict() if title_kwargs is None else title_kwargs
    ttl = ax.set_title(title, **title_kwargs) if title else None
    if xlabel:
        xlabel_kwargs = dict() if xlabel_kwargs is None else xlabel_kwargs
        xttl = ax.set_xlabel(xlabel, **xlabel_kwargs)
    return scat, hl, ttl, xttl


def _highlight_kwargs_defaults(highlight_kwargs:dict)->dict:
    """Sets some defaults for highlighting kwargs in statistical discriminator plot"""
    highlight_kwargs = dict() if highlight_kwargs is None else highlight_kwargs
    highlight_kwargs.setdefault('marker', '*')
    highlight_kwargs.setdefault('s', 80)
    if 'color' not in highlight_kwargs and 'c' not in highlight_kwargs:
        highlight_kwargs['c'] = 'r'
    return highlight_kwargs


def scatter_BIC(data:PhotonDataS, statepaths:Sequence[Param], ax:plt.Axes=None,
                thresh:float=0.005, highlight_kwargs:dict[str:Any]=None, 
                **kwargs:Any)->tuple[mpl.collections.PatchCollection,mpl.collections.PatchCollection,plt.Text,plt.Text]:
    """
    Plot BIC of a sequence of statepaths.
    
    General patter of use:
    
    .. code-block::
        
        statepaths = bhm.StatePath.optimize_models(data, to_state=4, max_state=8)
        bhm.plot.scatter_BIC(data, statepaths)

    .. Note::
        
        This is a wrapper with some basic defaults of :func:`scatter_model_statdisc`

    Parameters
    ----------
    data : PhotonDataS
        Data on which statepaths are based 
        (use this as source of data to compute the BIC).
    statepaths : Sequence[Param]
        Sequence of |StatePath| based ``Param`` defining the models for which the
        BIC should be computed and plotted, generally each should be of a different 
        number of states but otherwise the same parameters.
    ax : plt.Axes, optional
        The |Axes| in which to place plot. If None, use current axes.
        The default is None.
    thresh : float, optional
        Threshold for highlighting a state as ideal. The default is 0.005.
    highlight_kwargs : dict[str:Any], optional
        Kwargs handed to |axscatter| for the ideal model. Some defaults automatically
        set if not supplied.
        The default is None.
    title : None|bool|str, optional
        Title to give to plot, if None or True, automatically set title based on 
        statistical discriminator, if False, do not set title, if str, use string
        directly as title. The default is "$BIC$".
    title_kwargs : dict[str:Any], optional
        Keyword arguments handed to |axtitle|. The default is None.
    xlabel : None|False|str, optional
        xlabel of plot, if :code:`None` or :code:`False` do not plot. The default is 'States'.
    xlabel_kwargs : dict[str:Any], optional
        Kwargs handed to |axxlabel|.
        The default is None.
    **kwargs : Any
        Additional kwargs handed to |axscatter| for plotting models BIC.

    Returns
    -------
    scat : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of non-highlighted states.
    hl : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of highlighted state.
    ttl : plt.Text | None
        |plttext| object of title.
    xttl : plt.Text | None
        |plttext| object of x-axis label.

    """
    return scatter_model_statdisc(data, statepaths, ax=ax, stat=calc_BIC, highlight=calc_BIC, thresh=thresh, 
                                  highlight_kwargs=_highlight_kwargs_defaults(highlight_kwargs), 
                                  **kwargs)


def scatter_BICph(data:PhotonDataS, statepaths:Sequence[Param], ax:plt.Axes=None,
                thresh:float=0.005, highlight_kwargs:dict[str:Any]=None, 
                **kwargs:Any)->tuple[mpl.collections.PatchCollection,mpl.collections.PatchCollection]:
    """
    Plot BIC per photon of a sequence of statepaths.
    
    General patter of use:
    
    .. code-block::
        
        statepaths = bhm.StatePath.optimize_models(data, to_state=4, max_state=8)
        bhm.plot.scatter_BICph(data, statepaths)


    .. Note::
        
        This is a wrapper with some basic defaults of :func:`scatter_model_statdisc`


    Parameters
    ----------
    data : PhotonDataS
        Data on which statepaths are based 
        (use this as source of data to compute the BIC per photon).
    statepaths : Sequence[Param]
        Sequence of |StatePath| based ``Param`` defining the models for which the
        BIC per photon should be computed and plotted, generally each should be of a different 
        number of states but otherwise the same parameters.
    ax : plt.Axes, optional
        The |Axes| in which to place plot. If None, use current axes.
        The default is None.
    thresh : float, optional
        Threshold for highlighting a state as ideal. The default is 0.005.
    highlight_kwargs : dict[str:Any], optional
        Kwargs handed to |axscatter| for the ideal model. Some defaults automatically
        set if not supplied.
        The default is None.
    title : None|bool|str, optional
        Title to give to plot, if None or True, automatically set title based on 
        statistical discriminator, if False, do not set title, if str, use string
        directly as title. The default is "$BIC ph^{-1}$".
    title_kwargs : dict[str:Any], optional
        Keyword arguments handed to |axtitle| . The default is None.
    xlabel : None|False|str, optional
        xlabel of plot, if :code:`None` or :code:`False` do not plot. The default is 'States'.
    xlabel_kwargs : dict[str:Any], optional
        Kwargs handed to |axxlabel|.
        The default is None.
    **kwargs : Any
        Additional kwargs handed to |axscatter| for plotting models BIC per photon.

    Returns
    -------
    scat : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of non-highlighted states.
    hl : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of highlighted state.
    ttl : plt.Text | None
        |plttext| object of title.
    xttl : plt.Text | None
        |plttext| object of x-axis label.

    """
    return scatter_model_statdisc(data, statepaths, ax=ax, stat=calc_BICph, highlight=calc_BICph, thresh=thresh, 
                                  highlight_kwargs=_highlight_kwargs_defaults(highlight_kwargs), 
                                  **kwargs)


def scatter_BICp(data:PhotonDataS, statepaths:Sequence[Param], ax:plt.Axes=None,
                thresh:float=0.005, highlight_kwargs:dict[str:Any]=None, 
                **kwargs:Any)->tuple[mpl.collections.PatchCollection,mpl.collections.PatchCollection,plt.Text,plt.Text]:
    """
    Plot modified BIC of a sequence of statepaths, (from )
    
    General patter of use:
    
    .. code-block::
        
        statepaths = bhm.StatePath.optimize_models(data, to_state=4, max_state=8)
        bhm.plot.scatter_BICp(data, statepaths)

    .. Note::
        
        This is a wrapper with some basic defaults of :func:`scatter_model_statdisc`

    Parameters
    ----------
    data : PhotonDataS
        Data on which statepaths are based 
        (use this as source of data to compute the modified BIC).
    statepaths : Sequence[Param]
        Sequence of |StatePath| based ``Param`` defining the models for which the
        modified BIC should be computed and plotted, generally each should be of a different 
        number of states but otherwise the same parameters.
    ax : plt.Axes, optional
        The |Axes| in which to place plot. If None, use current axes.
        The default is None.
    thresh : float, optional
        Threshold for highlighting a state as ideal. The default is 0.005.
    highlight_kwargs : dict[str:Any], optional
        Kwargs handed to |axscatter| for the ideal model. Some defaults automatically
        set if not supplied.
        The default is None.
    title : None|bool|str, optional
        Title to give to plot, if None or True, automatically set title based on 
        statistical discriminator, if False, do not set title, if str, use string
        directly as title. The default is r"$BIC^{\prime}$".
    title_kwargs : dict[str:Any], optional
        Keyword arguments handed to |axtitle|. The default is None.
    xlabel : None|False|str, optional
        xlabel of plot, if :code:`None` or :code:`False` do not plot. The default is 'States'.
    xlabel_kwargs : dict[str:Any], optional
        Kwargs handed to |axxlabel|.
        The default is None.
    **kwargs : Any
        Additional kwargs handed to |axscatter| for plotting models modified BIC.

    Returns
    -------
    scat : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of non-highlighted states.
    hl : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of highlighted state.
    ttl : plt.Text | None
        |plttext| object of title.
    xttl : plt.Text | None
        |plttext| object of x-axis label.

    """
    return scatter_model_statdisc(data, statepaths, ax=ax, stat=calc_BICp, highlight=calc_BICp, thresh=thresh, 
                                  highlight_kwargs=_highlight_kwargs_defaults(highlight_kwargs), 
                                  **kwargs)


def scatter_ICL(data:PhotonDataS, statepaths:Sequence[Param], ax:plt.Axes=None,
                thresh:float=0.0, highlight_kwargs:dict[str:Any]=None, 
                **kwargs:Any)->tuple[mpl.collections.PatchCollection,mpl.collections.PatchCollection,plt.Text,plt.Text]:
    """
    Plot ICL of a sequence of statepaths.
    
    General patter of use:
    
    .. code-block::
        
        statepaths = bhm.StatePath.optimize_models(data, to_state=4, max_state=8)
        bhm.plot.scatter_ICL(data, statepaths)

    .. Note::
        
        This is a wrapper with some basic defaults of :func:`scatter_model_statdisc`

    Parameters
    ----------
    data : PhotonDataS
        Data on which statepaths are based 
        (use this as source of data to compute the ICL).
    statepaths : Sequence[Param]
        Sequence of |StatePath| based ``Param`` defining the models for which the
        ICL should be computed and plotted, generally each should be of a different 
        number of states but otherwise the same parameters.
    ax : plt.Axes, optional
        The |Axes| in which to place plot. If None, use current axes.
        The default is None.
    thresh : float, optional
        Threshold for highlighting a state as ideal. The default is 0.005.
    highlight_kwargs : dict[str:Any], optional
        Kwargs handed to |axscatter| for the ideal model.
        Some defaults automatically set if not supplied.
        The default is None.
    title : None|bool|str, optional
        Title to give to plot, if None or True, automatically set title based on 
        statistical discriminator, if False, do not set title, if str, use string
        directly as title. The default is "$ICL$".
    title_kwargs : dict[str:Any], optional
        Keyword arguments handed to |axtitle|. The default is None.
    xlabel : None|False|str, optional
        xlabel of plot, if :code:`None` or :code:`False` do not plot. The default is 'States'.
    xlabel_kwargs : dict[str:Any], optional
        Kwargs handed to |axxlabel|.
        The default is None.
    **kwargs : Any
        Additional kwargs handed to |axscatter| for plotting models ICL.

    Returns
    -------
    scat : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of non-highlighted states.
    hl : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of highlighted state.
    ttl : plt.Text | None
        |plttext| object of title.
    xttl : plt.Text | None
        |plttext| object of x-axis label.

    """
    return scatter_model_statdisc(data, statepaths, ax=ax, stat=calc_ICL, highlight=calc_ICL, thresh=thresh, 
                                  highlight_kwargs=_highlight_kwargs_defaults(highlight_kwargs), 
                                  **kwargs)

def scatter_ICLph(data:PhotonDataS, statepaths:Sequence[Param], ax:plt.Axes=None,
                thresh:float=0.0, highlight_kwargs:dict[str:Any]=None, 
                **kwargs:Any)->tuple[mpl.collections.PatchCollection,mpl.collections.PatchCollection]:
    """
    Plot ICL per photon of a sequence of statepaths.
    
    General patter of use:
    
    .. code-block::
        
        statepaths = bhm.StatePath.optimize_models(data, to_state=4, max_state=8)
        bhm.plot.scatter_ICLph(data, statepaths)

    .. Note::
        
        This is a wrapper with some basic defaults of :func:`scatter_model_statdisc`

    Parameters
    ----------
    data : PhotonDataS
        Data on which statepaths are based 
        (use this as source of data to compute the ICL per photon).
    statepaths : Sequence[Param]
        Sequence of |StatePath| based ``Param`` defining the models for which the
        ICL per photon should be computed and plotted, generally each should be of a different 
        number of states but otherwise the same parameters.
    ax : plt.Axes, optional
        The |Axes| in which to place plot. If None, use current axes.
        The default is None.
    thresh : float, optional
        Threshold for highlighting a state as ideal. The default is 0.005.
    highlight_kwargs : dict[str:Any], optional
        Kwargs handed to |axscatter| for the ideal model. Some defaults automatically
        set if not supplied.
        The default is None.
    title : None|bool|str, optional
        Title to give to plot, if None or True, automatically set title based on 
        statistical discriminator, if False, do not set title, if str, use string
        directly as title. The default is "$ICLph^{-1}$".
    title_kwargs : dict[str:Any], optional
        Keyword arguments handed to |axtitle|. The default is None.
    xlabel : None|False|str, optional
        xlabel of plot, if :code:`None` or :code:`False` do not plot. The default is 'States'.
    xlabel_kwargs : dict[str:Any], optional
        Kwargs handed to |axxlabel|.
        The default is None.
    **kwargs : Any
        Additional kwargs handed to |axscatter| for plotting models ICL per photon.

    Returns
    -------
    scat : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of non-highlighted states.
    hl : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of highlighted state.
    ttl : plt.Text | None
        |plttext| object of title.
    xttl : plt.Text | None
        |plttext| object of x-axis label.

    """
    return scatter_model_statdisc(data, statepaths, ax=ax, stat=calc_ICLph, highlight=calc_ICLph, thresh=thresh, 
                                  highlight_kwargs=_highlight_kwargs_defaults(highlight_kwargs), 
                                  **kwargs)


def scatter_pathBIC(data:PhotonDataS, statepaths:Sequence[Param], ax:plt.Axes=None,
                thresh:float=0.0, highlight_kwargs:dict[str:Any]=None, 
                **kwargs:Any)->tuple[mpl.collections.PatchCollection,mpl.collections.PatchCollection,plt.Text,plt.Text]:
    """
    Plot BIC of most likely statepath of a sequence of statepaths.
    
    General patter of use:
    
    .. code-block::
        
        statepaths = bhm.StatePath.optimize_models(data, to_state=4, max_state=8)
        bhm.plot.scatter_pathBIC(data, statepaths)

    .. Note::
        
        This is a wrapper with some basic defaults of :func:`scatter_model_statdisc`

    Parameters
    ----------
    data : PhotonDataS
        Data on which statepaths are based 
        (use this as source of data to compute the BIC of most likely state path).
    statepaths : Sequence[Param]
        Sequence of |StatePath| based ``Param`` defining the models for which the
        BIC of most likely state path should be computed and plotted, 
        generally each should be of a different number of states,
        but otherwise the same parameters.
    ax : plt.Axes, optional
        The |Axes| in which to place plot. If None, use current axes.
        The default is None.
    thresh : float, optional
        Threshold for highlighting a state as ideal. The default is 0.005.
    highlight_kwargs : dict[str:Any], optional
        Kwargs handed to |axscatter| for the ideal model. Some defaults automatically
        set if not supplied.
        The default is None.
    title : None|bool|str, optional
        Title to give to plot, if None or True, automatically set title based on 
        statistical discriminator, if False, do not set title, if str, use string
        directly as title. The default is "$BIC_{path}$".
    title_kwargs : dict[str:Any], optional
        Keyword arguments handed to |axtitle|. The default is None.
    xlabel : None|False|str, optional
        xlabel of plot, if :code:`None` or :code:`False` do not plot. The default is 'States'.
    xlabel_kwargs : dict[str:Any], optional
        Kwargs handed to |axxlabel|.
        The default is None.
    **kwargs : Any
        Additional kwargs handed to |axscatter| for plotting models BIC of most
        likely state path.

    Returns
    -------
    scat : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of non-highlighted states.
    hl : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of highlighted state.
    ttl : plt.Text | None
        |plttext| object of title.
    xttl : plt.Text | None
        |plttext| object of x-axis label.

    """
    return scatter_model_statdisc(data, statepaths, ax=ax, stat=calc_pathBIC, highlight=calc_pathBIC, thresh=thresh, 
                                  highlight_kwargs=_highlight_kwargs_defaults(highlight_kwargs), 
                                  **kwargs)


def scatter_pathBICph(data:PhotonDataS, statepaths:Sequence[Param], ax:plt.Axes=None,
                thresh:float=0.0, highlight_kwargs:dict[str:Any]=None, 
                **kwargs:Any)->tuple[mpl.collections.PatchCollection,mpl.collections.PatchCollection,plt.Text,plt.Text]:
    """
    Plot BIC of most likely statepath per photon of a sequence of statepaths.
    
    General patter of use:
    
    .. code-block::
        
        statepaths = bhm.StatePath.optimize_models(data, to_state=4, max_state=8)
        bhm.plot.scatter_pathBICph(data, statepaths)


    .. Note::
        
        This is a wrapper with some basic defaults of :func:`scatter_model_statdisc`


    Parameters
    ----------
    data : PhotonDataS
        Data on which statepaths are based 
        (use this as source of data to compute the BIC of most likely state path
        per photon).
    statepaths : Sequence[Param]
        Sequence of |StatePath| based ``Param`` defining the models for which the
        BIC of most likely state path per photonshould be computed and plotted, 
        generally each should be of a different number of states,
        but otherwise the same parameters.
    ax : plt.Axes, optional
        The |Axes| in which to place plot. If None, use current axes.
        The default is None.
    thresh : float, optional
        Threshold for highlighting a state as ideal. The default is 0.005.
    highlight_kwargs : dict[str:Any], optional
        Kwargs handed to |axscatter| for the ideal model. Some defaults automatically
        set if not supplied.
        The default is None.
    title : None|bool|str, optional
        Title to give to plot, if None or True, automatically set title based on 
        statistical discriminator, if False, do not set title, if str, use string
        directly as title. The default is "$BIC_{path} ph^{-1}$".
    title_kwargs : dict[str:Any], optional
        Keyword arguments handed to |axtitle|. The default is None.
    xlabel : None|False|str, optional
        xlabel of plot, if :code:`None` or :code:`False` do not plot. The default is 'States'.
    xlabel_kwargs : dict[str:Any], optional
        Kwargs handed to |axxlabel|.
        The default is None.
    **kwargs : Any
        Additional kwargs handed to |axscatter| for plotting models BIC of most
        likely state path per photon.

    Returns
    -------
    scat : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of non-highlighted states.
    hl : mpl.collections.PatchCollection
        |PathCollection| of scatter plot of highlighted state.
    ttl : plt.Text | None
        |plttext| object of title.
    xttl : plt.Text | None
        |plttext| object of x-axis label.

    """
    return scatter_model_statdisc(data, statepaths, ax=ax, stat=calc_pathBICph, highlight=calc_pathBICph, thresh=thresh, 
                                  highlight_kwargs=_highlight_kwargs_defaults(highlight_kwargs), 
                                  **kwargs)

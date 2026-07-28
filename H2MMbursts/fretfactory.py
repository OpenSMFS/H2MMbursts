#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience functions for creating FRET ALEX/PIE based params etc.

.. |StatePath| replace:: :class:`StatePath <H2MMbursts.modeltables.StatePath>`
.. |StatePathBase| replace:: :class:`StatePathBase <H2MMbursts.modeltables.StatePathBase>`
.. |Dwells| replace:: :class:`Dwells <H2MMbursts.modeltables.Dwells>`
.. |Param| replace:: :class:`Param <smfbursts.datamodel.tables.Param>`
.. |Column| replace:: :class:`Column <smfbursts.datamodel.tables.Column>`
.. |DetDef| replace:: :class:`DetDef <smfbursts.ph_sel.DetDef>`
.. |pdstyler| replace:: `pandas.io.formats.style.Styler <https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.html>`__
"""
from numbers import Real, Integral
from itertools import repeat
from collections.abc import Sequence

import numpy as np
from scipy.stats import gmean
import pandas as pd


from smfbursts import rcParams
from smfbursts.datamodel.tables import Param, Column
from smfbursts.photondata import PhotonData, PhotonDataS
from smfbursts.ph_sel import PhSel
from smfbursts.fretfactory import make_fret_from_base
import H2MM_C as hm

from .modeltables import Dwells


def _get_column(data:PhotonDataS, col:Column, record:bool|None=None)->np.ndarray:
    record = rcParams['plot.record'] if record is None else record
    val = data.record_column(col) if record else data.get_column(col)
    if isinstance(val, tuple):
        val = np.concatenate(val)
    return val


def make_dwell_dict(statepath:Param, include_statepath:bool=True, update:dict=None, 
                    **kwargs)->dict[str:Column|Param]:
    """
    Build a dictionary of default |Dwells| based |Column| objects from an input
    |StatePathBase| based |Param|.
    
    This function wraps :func:`smfbursts.fretfactory.make_fret_from_base`,
    handing it the |Dwells| based |Param| generated from the |StatePathBase|
    based |Param|.
    
    It adds the following keys:
        
        - 'dwells': A |Dwells| based |Param| with ``statepath`` as its parent
        - 'State': The state of each dwell
    
    If ``include_statepath`` is :code:`True`, then the following keys are also
    added:
        
        - 'statepath': the input |Param|
        - 'Bstate': the 'bstates' |Column| of ``statepath``, the state-code of
          the bursts.

    Parameters
    ----------
    statepath : Param
        |StatePathBase| based |Param| to build a dictionary of |Dwell| based |Columns|.
    include_statepath : bool, optional
        Whether or not to add the |StatePathBase| based |Param| to the dictionary
        under the key 'statepath'. The default is True.
    update : dict, optional
        Dictionary to update with the new columns. The default is None.
    **kwargs : Any
        Additional kwargs handed to :func:`smfbursts.fretfactory.make_fret_from_base`.

    Returns
    -------
    dict[str:Column | Param]
        Dictionary of |Dwells| based |Column|.

    """
    dwells = Param(Dwells, statepath=statepath)
    update = make_fret_from_base(dwells, update=update, **kwargs)
    update['dwells'] = dwells
    update['State'] = Column(dwells, 'state')
    if include_statepath:
        update['Bstates'] = Column(statepath, 'bstates')
        update['statepath'] = statepath
    return update


def make_usALEX_trans_limits(data:PhotonDataS, factor:float=10.0)->hm.h2mm_limits:
    r"""
    Generate a limits object to bound transition rates of an the optimization 
    of a :math:`\mathrm{\mu sALEX}` data set to avoid artefacts from the
    laser alternation rate.
    

    Parameters
    ----------
    data : PhotonDataS
        Photon data defining the alternation rate (should be usALEX experiment(s)).
    factor : float, optional
        Fraction of alternation rate to multiply to get maximum transition rate. 
        The default is 10.0.

    Returns
    -------
    hm.h2mm_limits
        Limits preventing transition rates from being too fast.

    """
    setup = data.setup if isinstance(data, PhotonData) else data.datas[0].setup
    return hm.h2mm_limits(max_trans=1.0/(factor*setup.alex_period))


def summary_frame(data:PhotonDataS, statepath:Param, ph_min:int=5, record:bool|None=None)->pd.DataFrame:
    """
    Create a summary dataframe of a |H2MM| model defined by a |StatePath| based
    |Param| and the underlying data.

    Parameters
    ----------
    data : PhotonDataS
        Source of data.
    statepath : Param
        |StatePathBase| based param defining the model for which to create a
        summary frame.
    ph_min : int, optional
        Minimum number of photons in a dwell to be considered a "true"
        (aka reliable) dwell. The default is 5.
    record : bool | None, optional
        Whether to record columns, if None, use default of smfbursts.
        The default is None.

    Returns
    -------
    pd.DataFrame
        Summary dataframe of statepath.

    """
    vals = list()
    bursts = statepath.parents['bursts']
    model = statepath.params['model']
    nstate = model.nstate
    trans = model.trans
    dwells = Param(Dwells, statepath=statepath)
    sel_span = statepath.tp.get_phsel_span(statepath)
    # check which parameters can be computed
    if all(sel in sel_span for sel in (PhSel('0ex0em'), PhSel('0ex1em'))):
        vals.append(('E h2mm', Column(bursts, 'E_raw')))
    if all(sel in sel_span for sel in (PhSel('0ex0em'), PhSel('0ex1em'), PhSel('1ex1em'))):
        vals.append(('S h2mm', Column(bursts, 'S_raw')))
    # compute values based on H2MM model
    hparams = statepath.model_values(*(c for _, c in vals), origin=data)
    hparams = {k:v for (k, _), v in zip(vals, hparams)}
    # create Dwells based Column objects for retrieving dwell populations
    dvals = [(_get_column(data, Column(dwells, 'nph_raw', c.keytup[0]), record), 
              data.get_column(Column(dwells, 'nph_raw', c.keytup[1])))
             for _, c in vals]
    dstate = _get_column(data, Column(dwells, 'state'), record)
    dnextstate = _get_column(data, Column(dwells, 'state', 1), record) # the next state in the sequence
    middwells = _get_column(data, Column(dwells, 'dwell_pos'), record) > 0 # if dwell_pos is > 0, dwell is middle dwell
    num_mid_dwells = np.zeros((model.nstate, model.nstate), dtype=np.int64)
    # allocate arrays for viterbi values
    vit_vals = [np.nan*np.ones(nstate, dtype=np.float64) for _ in range(len(vals))]
    vit_err = [np.nan*np.ones(nstate, dtype=np.float64) for _ in range(len(vals))]
    # compute mean and std of each state by wells
    states = dict()
    for s in range(nstate):
        states[f'to state {s}'] = trans[:,s]/data.clk_p
        mask = dstate == s
        num_mid_dwells[:,s] = np.bincount(dstate[middwells&(dnextstate==s)], minlength=nstate)
        if not np.any(mask):
            continue
        for dw, err, (num, dem) in zip(vit_vals, vit_err, dvals):
            zmask = mask & (dem >= ph_min)
            n, d = num[zmask], dem[zmask]
            dsum = d.sum()
            dw[s] = n.sum() / dsum
            err[s] = np.sqrt(np.sum(d*((n/d - dw[s])**2))/dsum)
    hparams.update({k.strip(' h2mm')+' vit':vit for (k, _), vit in zip(vals, vit_vals)})
    hparams.update({k.strip(' h2mm')+' err':err for (k, _), err in zip(vals, vit_err)})
    hparams.update(states)
    hparams.update({f'num mid transistion to state {i}':num_mid_dwells[:,i] for i in range(nstate)})
    return pd.DataFrame(hparams)


def highlight_rateframe(frame:pd.DataFrame, data:PhotonDataS=None, statepath:Param=None, 
                        max_rate:float=None, min_rate:float=None, min_dwells:int=10, 
                        fast_fmt:str='color: red', valid_fmt:str='color: blue', 
                        slow_fmt:str='color: purple', bad_fmt:str='color: orange', 
                        max_rate_factor:float=10.0, slow_rate_factor:float=10.0, 
                        record:bool|None=None):
    r"""
    Generate a |pdstyler| to display the output of :func:`summary_frame` with
    cells of transition rates highlighted according to their reasonableness.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of :func:`summary_frame`.
    data : PhotonDataS, optional
        Source of data for |H2MM| optimization. The default is None.
    statepath : Param, optional
        |StatePath| based |Param| specifying the |H2MM| model for which frame
        was generated. The default is None.
    max_rate : float, optional
        Rate (in s) above which to highlight with fast_fmt, ie transition rate
        too fast. If None, compute the maximum transition rate as 
        :math:`max\_rate\_factor * geomean(max\_rate)`
        The default is None.
    min_rate : float, optional
        Rate (in s) below which to highlight with slow_fmt, ie transition rate
        too slow. If None, compute the minimum transition rate as 
        :math:`min\_rate\_factor * geomean(burst\_duration)`
        The default is None.
    min_dwells : int, optional
        Minimum number of from->to dwells for transition rate to not be marked
        as "bad". The default is 10.
    fast_fmt : str, optional
        Format option for cells . The default is 'color: red'.
    valid_fmt : str, optional
        Format option for transition rates that are "good" 
        i.e. between min_rate and max_rate
        and ma. 
        The default is 'color: blue'.
    slow_fmt : str, optional
        Format option for transision rates below min_rate. 
        The default is 'color: purple'.
    bad_fmt : str, optional
        Format option of cells with fewer than min_dwells dwells. 
        The default is 'color: orange'.
    max_rate_factor : float, optional
        Factor by which to multiply geometric mean of max photon rate of bursts
        to compute min_rate. The default is None.
    slow_rate_factor : float, optional
        Factor by which to multiply geometric mean of duration of bursts to
        compute min_rate. The default is None.
    record : bool|None, optional
        Whether to try to save computed columns in cache. The default is None.

    Raises
    ------
    ValueError
        data and statepath not specified when using automatic max/min_rate
        determination.

    Returns
    -------
    pd.io.formats.style.Styler
        |pdstyler| which displays input dataframe with rates highlighted for
        above/bellow max/min rate, and for having too few dwells.

    """
    if max_rate is None:
        if data is None or statepath is None:
            raise ValueError("must specify data and statepath if max_rate is not specified")
        max_rate_factor = 10.0 if max_rate_factor is None else max_rate_factor
        mrparam = Column(statepath.parents['bursts'], 'max_rate', (PhSel('0ex'), 10))
        max_rate = gmean(_get_column(data, mrparam)) / max_rate_factor
    if min_rate is None:
        if data is None or statepath is None:
            raise ValueError("must specify data and statepath if min_rate is not specified")
        durparam = Column(statepath.parents['bursts'], 'dur', ('istarttime', 'istoptime'))
        slow_rate_factor = 10.0 if slow_rate_factor is None else slow_rate_factor
        min_rate = gmean(_get_column(data, durparam, record)) * slow_rate_factor
    def highlight_cell(trate:Real, ndwell:Real)->str:
        """Returns highligh for a cell with value trate, and number of dwells (from other column) ndwell"""
        if trate > max_rate:
            return fast_fmt
        if trate < min_rate:
            return 
        return bad_fmt if ndwell < min_dwells else valid_fmt
    
    def highlight_styler(row:str)->list[str]:
        """Highlight function for style.apply"""
        m = row.name.split(' ')[-1]
        nd = frame[f'num mid transistion to state {m}']
        return [highlight_cell(r, n) for r, n in zip(row, nd)]
    
    return frame.style.apply(highlight_styler, subset=[c for c in frame.columns if c.startswith('to state')])


def _make_divisor(data:PhotonDataS, bursts:Param, stream:PhSel, ndiv:int, irf_thresh:bool)->np.ndarray[np.uint16]:
    qtile = np.linspace(0.0, 1.0, ndiv+2)[1:-1]
    if bursts is None:
        nanos = data.get_nanos(stream)
        nanos = nanos if isinstance(data, PhotonData) else np.concatenate(nanos)
    else:
        col = Column(bursts, 'ph_nanos', (stream))
        nanos = data.get_column(col) if isinstance(data, PhotonData) else data.concatenate_column(col)
        nanos = np.concatenate(nanos)
    if irf_thresh:
        nanos = nanos[nanos >= data.irf_thresh[stream]]
    divs = np.quantile(nanos, qtile).astype(nanos.dtype)
    if irf_thresh:
        divs = np.concatenate([[data.irf_thresh[stream]], divs])
    return divs
    

def make_divisors(data:PhotonDataS, bursts:Param=None, streams:Sequence[PhSel]=None, 
                  ndivs:int|Sequence[int]=1, include_irf_thresh:bool|Sequence[bool]=False
                  )->tuple[np.ndarray[np.uint16],...]:
    """
    Generate divisors that evenly divide the data of the given stream based
    on frequency of nanotimes.

    Parameters
    ----------
    data : PhotonDataS
        Data for which to create evenly distributed divisors based on nanotimes.
    bursts : Param, optional
        Burst selection (including gate) defining which nanotimes to use,
        if not specified, then take all nanotimes of entire dataset, no background
        filtering. The default is None.
    streams : Sequence[PhSel], optional
        Sequence of streams for which to generate divisors. If None, assume
        1 stream for each detector index. The default is None.
    ndivs : int|Sequence[int], optional
        Number of divisors per stream. If given as single int, set same number
        of divisors in each stream in streams. The default is 1.
    include_irf_thresh : bool|Sequence[bool], optional
        Whether to add a divisor for the irf_thresh. Note if true, that this results
        in the division of even divisors now only using photons with nanotimes
        over the irf_thresh, and that the number of divisors is +1 from the
        number of divisors specified in ndivs. Can be specified as single
        bool for all streams, or as sequence of bools, 1 for each stream.
        The default is False.

    Returns
    -------
    tuple[np.ndarray[np.uint16,...]]
        Divisor arrays, able to be used as the "divs" parameter of a
        :class:`H2MMbursts.modeltables.ntdivStatePath` based |Param|.

    """
    if streams is None:
        streams = tuple(data.detdef.stream_ids_to_PhSel(i, convert_all=True) 
                        for i in range(data.detdef.size))
    streams = (streams, ) if isinstance(streams, PhSel) else streams
    ndivs = 1 if ndivs is None else ndivs
    ndivs = repeat(ndivs) if isinstance(ndivs, Integral) else ndivs
    irf_thresh = repeat(include_irf_thresh) if isinstance(include_irf_thresh, bool) else include_irf_thresh
    return tuple( _make_divisor(data, bursts, stream, ndiv, irf_t) 
                 for stream, ndiv, irf_t in zip(streams, ndivs, irf_thresh))

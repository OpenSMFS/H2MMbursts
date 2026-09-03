#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 10:59:48 2026

@author: paul
"""
from sys import version_info as python_version
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version('H2MMbursts')
except PackageNotFoundError:
    print("Cannot find package version")
    __version__ = 'undefined'
del python_version, version, PackageNotFoundError


from . import _citations

from .modeltables import StatePath, Dwells, ntdivStatePath, usAlexStatePath, StatePathFilter
from . import erroranalysis as error
from . import simulations as sim
from . import fretfactory

from H2MM_C import h2mm_model, factory_h2mm_model, optimization_limits

from smfbursts.datamodel import has_matplotlib
if has_matplotlib:
    from . import plot


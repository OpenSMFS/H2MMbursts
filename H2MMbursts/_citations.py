# -*- coding: utf-8 -*-
# Author : Paul David Harris
# email : harripd@gmail.com
# Created : 11/12/2025
"""
Load citations for H2MMbursts

.. note::
    
    This script requires that all bibliographic fields have a DOI
"""
from importlib.resources import files
import json

from smfbursts.cite import register_citation, create_citation_group
from smfbursts._citations import _doi_bib, _split_ris, _doi_ris


_bibtex = {_doi_bib(cite):f'@{cite}' for cite in 
           files('H2MMbursts').joinpath('citations/citations.bib').read_text(encoding='utf8').split('@') 
           if cite}
    
_risrefs = {_doi_ris(record):record for record in 
            _split_ris(files('H2MMbursts').joinpath('citations/citations.ris').read_text(encoding='utf8')) 
            if record}

_citations = json.loads(files('H2MMbursts').joinpath('citations/citations.json').read_text(encoding='utf8'))

_styles = {'bibtex':_bibtex, 'ris':_risrefs}


def _cite_kwargs(doi:str)->dict[str,str]:
    """Create kwargs of all styles speficied for given doi"""
    out = dict()
    if not doi.startswith('X'):
        out['doi'] = doi
    for style, records in _styles.items():
        if doi in records:
            out[style] = records[doi]
    return out

_bhmcite = tuple(register_citation(tag, citation, **_cite_kwargs(doi)) 
                 for tag, (citation, doi) in _citations.items())

H2MMbursts_citations = tuple(f.citation for f in _bhmcite)

create_citation_group('h2mm', 'PirchiJPCB2016', 'HarrisNatComms2022')
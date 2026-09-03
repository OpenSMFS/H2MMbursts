# Author: Paul David Harris
# Created: 04/05/2026
# Purpose: Download files from Zenodo repository
import pathlib
import pooch
import zipfile

DATASET_DIR = u'data'

repo = pooch.create(path=DATASET_DIR, base_url='doi:10.5281/zenodo.20038738')
repo.load_registry_from_doi()

files = ('HP3_TE300_SPC630.hdf5', 
         )

for file in files:
    repo.fetch(file)

repo = pooch.create(path=DATASET_DIR, base_url='doi:10.5281/zenodo.5902313')
repo.load_registry_from_doi()

files = ('mpH2MM_YopO_100us.zip', )

for file in files:
    repo.fetch(file)

with zipfile.ZipFile(pathlib.Path(DATASET_DIR) / files[0], 'r') as zp:
    zp.extractall(pathlib.Path(DATASET_DIR))

# 3Drec  
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)  
 
3Drec provides the implementation of two deep learning networks designed to reconstruct the 3D ocean structure from 2D satellite data: one based on a deep feed-forward network (**FFNN3D**) and one based on a stacked Long-Short Term Memory network (**LSTM3D**).  

The two networks and the training/test data used for their development are fully described in the paper:  
- Buongiorno Nardelli, B., A Deep Learning Network to Retrieve Ocean Hydrographic Profiles from Combined Satellite and In Situ Measurements, _Remote Sensing_, **2020**. [![DOI:-](https://zenodo.org/badge/DOI/-.svg)](https://doi.org/-)  

## Installation

The code is written in Python 3
  
These are the python packages required (tested versions):  
- keras     2.2.4
- numpy     1.18.1
- netcdf4   1.5.3
- pandas    1.0.3 
- seawater  3.3.4  

## Training data
The data used to develop the models can be found at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4040843.svg)](https://doi.org/10.5281/zenodo.4040843)

### Author
Bruno Buongiorno Nardelli, Consiglio Nazionale delle Ricerche - Istituto di Scienze Marine
<div itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0002-3416-7189" href="https://orcid.org/0000-0002-3416-7189" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">https://orcid.org/0000-0002-3416-7189</a></div>

### Funding
The development of this code was partly funded by the European Space Agency through the [World Ocean Current](https://www.worldoceancirculation.org) project (_ESA Contract No. 4000130730/20/I-NB_).

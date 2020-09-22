# 3Drec  
3Drec provides the implementation of two deep learning networks designed to reconstruct the 3D ocean structure from 2D satellite data: one based on a deep feed-forward network (**FFNN3D**) and one based on a stacked Long-Short Term Memory network (**LSTM3D**).  

The two networks and the training/test data used for their development are fully described in the paper:  
_Buongiorno Nardelli, A Deep Learning Network to Retrieve Ocean Hydrographic Profiles from Combined Satellite and In Situ Measurements, Remote Sensing, 2020_, under revision.  

The development of this code was partly funded by the European Space Agency through the World Ocean Current (**WOC**) project (_ESA Contract No. 4000130730/20/I-NB_).

## Installation
The code is written in Python 3
  
These are required python packages (tested versions):  
- keras     2.2.4
- numpy     1.18.1
- netcdf4   1.5.3
- pandas    1.0.3 
- seawater  3.3.4  

## Training data
The data used to develop the models can be found here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4040843.svg)](https://doi.org/10.5281/zenodo.4040843)

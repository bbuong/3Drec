#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a deep feed-forward network to retrieve ocean hydrographic profiles from combined satellite and in situ measurements

Reference: doi:........
@author: Bruno Buongiorno Nardelli

Consiglio Nazionale delle Ricerche
Istituto di Scienze Marine
Napoli, Italia

"""

import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import matplotlib.pyplot as plt
import numpy as np
from subprocess import call
import warnings

warnings.filterwarnings("ignore")  # specify to ignore warning messages
from keras.optimizers import SGD
from numpy import hstack
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from pandas import DataFrame
from keras.layers import TimeDistributed
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers.core import Lambda
from keras import backend as K
import scipy.io as sio
from netCDF4 import Dataset
from netCDF4 import getlibversion
import seawater as sw
import glob
from keras.models import load_model

def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))


def read_nc(netcdf_file):
    ncid = Dataset(netcdf_file, 'r')

    nc_vars = [var for var in ncid.variables]
    for var in nc_vars:
        if hasattr(ncid.variables[str(var)], 'add_offset'):
            exec('global ' + str(var) + "; offset=ncid.variables['" + str(var) + "'].add_offset; " + str(
                var) + "=ncid.variables['" + str(var) + "'][:]-offset")
        else:
            exec('global ' + str(var) + '; ' + str(var) + "=ncid.variables['" + str(var) + "'][:]")
    ncid.close()
    return

read_nc('insitu_test_clim.nc')
T_clim_test=T[:,:]
S_clim_test=S[:,:]
SH_clim_test=SH[:,:]

read_nc('surface_test_adj.nc')
tos_test=tos-np.repeat(T_clim_test[:,0][:, np.newaxis],tos.shape[1],axis=1)
adt_test=adt-np.repeat(SH_clim_test[:,0][:, np.newaxis],tos.shape[1],axis=1)
sos_test=sos-np.repeat(S_clim_test[:,0][:, np.newaxis],tos.shape[1],axis=1)


read_nc('insitu_test_qc.nc')

T_test=T[:,:]
S_test=S[:,:]
juld_pro_test=juld_pro[:]
latitude_pro_test=latitude_pro[:]
longitude_pro_test=longitude_pro[:]
juld_abs_test=juld_abs[:]

read_nc('insitu_training_clim.nc')
T_clim_training=T[:,:]
S_clim_training=S[:,:]
SH_clim_training=SH[:,:]

read_nc('insitu_training_qc.nc')

read_nc('surface_training_adj.nc')
tos=tos-np.repeat(T_clim_training[:,0][:, np.newaxis],tos.shape[1],axis=1)
adt=adt-np.repeat(SH_clim_training[:,0][:, np.newaxis],tos.shape[1],axis=1)
sos=sos-np.repeat(S_clim_training[:,0][:, np.newaxis],tos.shape[1],axis=1)


####################################
# pre-process training data
####################################

P=np.zeros(T.shape)
delta_P=10.
for i in range(P.shape[0]):
    P[i,:]=depth

D= sw.pden(S,T,P,pr=0)   #computes density profiles from in situ T and S
D_std=sw.pden(S*0+35,T*0,P,pr=0)  #computes standard density profiles
SVA= (1./D)         #computes specific volume
SVA_std = (1./D_std) #computes specific volume standard profiles
g=9.81

SH=np.zeros(T.shape)
for ik in range(T.shape[1]):
    SH[:,ik]=1e6*np.sum(SVA[:,ik:T.shape[1]],axis=1)*delta_P/g #steric heights in cm
    SH[:,ik]=SH[:,ik]-1e6*np.sum(SVA_std[:,ik:T.shape[1]],axis=1)*delta_P/g

T=T-T_clim_training
S=S-S_clim_training
SH=SH-SH_clim_training

################################
#  pre-process test data
################################

P_test = np.zeros(T_test.shape)
delta_P = 10.
for i in range(P_test.shape[0]):
    P_test[i, :] = depth

D_test = sw.pden(S_test, T_test, P_test, pr=0)  # computes density profiles from in situ T and S
D_std_test = sw.pden(S_test * 0 + 35, T_test * 0, P_test, pr=0)  # computes standard density profiles
SVA_test = (1. / D_test)  # computes specific volume
SVA_std_test = (1. / D_std_test)  # computes specific volume standard profiles
g = 9.81       

SH_test = np.zeros(T_test.shape)

for ik in range(T_test.shape[1]):
    SH_test[:, ik] = 1e6 * np.sum(SVA_test[:, ik:T_test.shape[1]], axis=1) * delta_P / g  # steric heights in cm
    SH_test[:, ik] = SH_test[:, ik] - 1e6 * np.sum(SVA_std_test[:, ik:T_test.shape[1]], axis=1) * delta_P / g

T_test=T_test-T_clim_test
S_test=S_test-S_clim_test
SH_test=SH_test-SH_clim_test


####################################
#FFNN model configuration parameters
####################################

activ = 'sigmoid'#'softsign'#
opt='Adam'
pat=30
n_epochs = 1000
val_split=.15
dropout_fraction=.2
n_units1 =1000
n_units2= 1000

# set input training variables

jd1=np.cos(2*np.pi*(juld_pro/365)+1)
jd2=np.sin(2*np.pi*(juld_pro/365)+1)

x0=tos-273.15
x1=sos
x2=adt
x3=np.zeros(x0.shape)
x4=np.zeros(x0.shape)
x5=np.zeros(x0.shape)
x6=np.zeros(x0.shape)

for ik in range(x0.shape[0]):
    x3[ik,:]=latitude_pro[ik]
    x4[ik,:]=longitude_pro[ik]
    x5[ik,:]= jd1[ik]
    x6[ik,:]= jd2[ik]


y0=SH
y1=T
y2=S


jd1_test = np.cos(2 * np.pi * (juld_pro_test / 365) + 1)
jd2_test = np.sin(2 * np.pi * (juld_pro_test / 365) + 1)

x0_test = tos_test-273.15
x1_test = sos_test
x2_test = adt_test
x3_test = np.zeros(x0_test.shape)
x4_test = np.zeros(x0_test.shape)
x5_test = np.zeros(x0_test.shape)
x6_test = np.zeros(x0_test.shape)

n_test = x0_test.shape[0]

for ik in range(x0_test.shape[0]):
    x3_test[ik, :] = latitude_pro_test[ik]
    x4_test[ik, :] = longitude_pro_test[ik]
    x5_test[ik, :] = jd1_test[ik]
    x6_test[ik, :] = jd2_test[ik]


y0_test = SH_test
y1_test = T_test
y2_test = S_test

##################################

label_y0='steric heights (cm)'
label_y1='temperature (°C)'
label_y2='salinity (mg/kg)'

n_depth = x0.shape[1]
n_samples= x0.shape[0]
n_steps_out = 1#fixed

check='check number of input variables'
i_var = 0
while not check=="stop":
    try:
        cmd = 'x'+str(i_var)
        exec(cmd)
    except NameError:
        n_var_in=i_var
        check="stop"
    i_var = i_var + 1

check='check number of output variables'
i_var = 0
while not check=="stop":
    try:
        cmd = 'y'+str(i_var)
        exec(cmd)
    except NameError:
        n_var_out=i_var
        check="stop"
    i_var = i_var + 1

print('number of output variables: ',n_var_out)

## Scale data

for i_var in range(n_var_in):
    cmd='xmax'+str(i_var)+'=x'+str(i_var)+'.max()'
    exec(cmd)
    cmd='xmin'+str(i_var)+'=x'+str(i_var)+'.min()'
    exec(cmd)
    cmd = 'xmax' + str(i_var) + '_test=x' + str(i_var) + '_test.max()'
    exec(cmd)
    cmd = 'xmin' + str(i_var) + '_test=x' + str(i_var) + '_test.min()'
    exec(cmd)
    cmd = 'xmax' + str(i_var) + '=np.max([xmax' + str(i_var) + ',xmax' + str(i_var) + '_test])'
    exec(cmd)
    cmd = 'xmin' + str(i_var) + '=np.min([xmin' + str(i_var) + ',xmin' + str(i_var) + '_test])'
    exec(cmd)

    cmd='xTrain'+str(i_var)+'=(x'+str(i_var)+'-xmin'+str(i_var)+')/(xmax'+str(i_var)+'-xmin'+str(i_var)+')'
    exec(cmd)

# xTrain0=x0/np.max(x0)

for i_var in range(n_var_out):
    cmd='ymax'+str(i_var)+'=y'+str(i_var)+'.max()'
    exec(cmd)
    cmd='ymin'+str(i_var)+'=y'+str(i_var)+'.min()'
    exec(cmd)
    cmd = 'ymax' + str(i_var) + '_test=y' + str(i_var) + '_test.max()'
    exec(cmd)
    cmd = 'ymin' + str(i_var) + '_test=y' + str(i_var) + '_test.min()'
    exec(cmd)
    cmd = 'ymax' + str(i_var) + '=np.max([ymax' + str(i_var) + ',ymax' + str(i_var) + '_test])'
    exec(cmd)
    cmd = 'ymin' + str(i_var) + '=np.min([ymin' + str(i_var) + ',ymin' + str(i_var) + '_test])'
    exec(cmd)


    cmd='yTrain'+str(i_var)+'=(y'+str(i_var)+'-ymin'+str(i_var)+')/(ymax'+str(i_var)+'-ymin'+str(i_var)+')'
    exec(cmd)

#prepare data for training input

X = np.zeros((n_samples, n_var_in))

for i_var in range(n_var_in):
    cmd = 'X[:,i_var]=xTrain' + str(i_var)+'[:,0]'
    exec(cmd)

cmd_str='yTrain0'
for i_var in range(1,n_var_out):
    cmd_str=cmd_str+',yTrain'+str(i_var)
cmd='y=hstack(('+cmd_str+'))'
exec(cmd)

##################################################################
#
#   model definition/fit
#
##################################################################

train = DataFrame()
val = DataFrame()
if not glob.glob('WOC_FFNN_sig_'+str(n_units1)+'_'+str(n_units2)+'MODEL.h5'):

    # define model
    model = Sequential()
    model.add(PermaDropout(dropout_fraction))
    model.add(Dense(n_units1, activation=activ, input_shape=(n_var_in,)))
    model.add(PermaDropout(dropout_fraction))
    if n_units2 > 0:
        model.add(Dense(n_units2, activation=activ))
        model.add(PermaDropout(dropout_fraction))
    model.add(Dense(n_var_out * n_depth))
    model.compile(loss='mse', optimizer=opt)

    # fit model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=pat)
    history=model.fit(X, y, epochs=n_epochs, verbose=1, shuffle=False, validation_split=val_split, callbacks=[es])


    model.save('WOC_FFNN_sig_'+str(n_units1)+'_'+str(n_units2)+'MODEL.h5')
    print("Saved model to disk")

    train = history.history['loss']
    val = history.history['val_loss']
    # plot train and validation loss across multiple runs
    plt.plot(train, color='blue', label='train')
    plt.plot(val, color='orange', label='validation')
    plt.title('FFNN ' + str(n_units1) + '-' + str(n_units2) + ' model')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show(block=False)
    plt.savefig('FFNN_sig_' + str(n_units1) + '_' + str(n_units2) + '_loss.eps', dpi=150)
    plt.close()
else:
    # load model
    model = load_model('WOC_FFNN_sig_'+str(n_units1)+'_'+str(n_units2)+'MODEL.h5')
    # summarize model.
    model.summary()


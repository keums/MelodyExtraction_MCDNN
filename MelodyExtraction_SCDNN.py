# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 15:15:44 2016

@author: keums
"""

import os
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from mySelect_weight import *

#plt.switch_backend('Qt4Agg')
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
np.random.seed(1337)  # for reproducibility                                                          
    
min_pitch = 38  # in MIDI                                                                                                                 
max_pitch = 78  # in MIDI    

def MelodyExtraction_SCDNN(x_test,note_res):
    '''
    predict melody using DNN model
        - layer : input-512-512-256-output
        - non-linear fuction : ReLU
        - optimizer = RMSprop
        - output_activation : sigmoid

    |Parameters|
    ----------  
    x_test: shape(features)
        a multi-frame specrogram of input .wav
    note_res:
        the number of resolution
    
    |Returns|
    -------
    y_predict: 
        a probability of melody extraction
    '''    
    dim_output = 40*note_res + 1 
       
    # DNN training
    model = Sequential()
    model.add(Dense(output_dim= 512, input_dim=x_test.shape[1], init='uniform'))
    model.add(Activation('relu')) #relu softplus
    model.add(Dropout(0.2))
    model.add(Dense(output_dim= 512, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim= 256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim = dim_output))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer = RMSprop())
    weight_file = mySelect_weight(note_res)

    model.load_weights(weight_file)
     # prediction
    y_predict = model.predict(x_test, batch_size=128, verbose=2)  
    print 'complete _res_'+str(note_res) +'_SCDNN'

    return y_predict
    
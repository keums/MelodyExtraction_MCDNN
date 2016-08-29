# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 18:42:17 2016

@author: keums
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from making_multi_frame_VAD import *

#plt.switch_backend('Qt4Agg')
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
np.random.seed(1337)  # for reproducibility                                                          
    
min_pitch = 38  # in MIDI                                                                                                                 
max_pitch = 78  # in MIDI    

def VAD_DNN(x_test,y_predict_in,param = 0.2 ):
    '''
    predict singing voice frame using DNN model
        - layer : input-512-256-128-output
        - non-linear fuction : ReLU
        - optimizer = RMSprop
        - output_activation : sigmoid

    |Parameters|
    ----------  
    x_test: shape(features)
        a single-frame specrogram of input .wav
    y_predict_in:
        the output of MCDNN(res=1) 
    
    |Returns|
    -------
    voice_frame: 
        a prediction of voice frame 
    '''            
    y_predict_SF = making_multi_frame_VAD(y_predict_in, num_frames= 1, max_bin = y_predict_in.shape[1], min_bin = 0)
    y_predict_SF = (y_predict_SF - y_predict_SF.mean(axis=0))/y_predict_SF.std(axis=0) 
    x_test_VAD = np.concatenate((y_predict_SF, x_test), axis=1)
       
    # DNN training
    model = Sequential()
    model.add(Dense(output_dim=512, input_dim=x_test_VAD.shape[1], init='uniform'))
    model.add(Activation('relu')) #relu softplus
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=256, init='uniform'))#512
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=128, init='uniform'))#256
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr = 0.001))

    weight_file  = './model/_frames_1_note_res_1_drop_out_0.2_optimizer_rms'+\
                        '_activation_relu_output_activation_sigmoid_SV.hdf5' 


    model.load_weights(weight_file)
     # prediction
    y_predict = model.predict(x_test_VAD, batch_size=128, verbose=2)

    voice_frame = (y_predict>param)
    voice_frame= voice_frame.astype(int)
    print 'complete VAD'
   
    return  voice_frame
    
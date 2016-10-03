# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 22:58:44 2016

@author: keums
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adadelta, SGD
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization

from load_train_data_RWC_dnn import *
from myModelCheckpoint import *

np.random.seed(1337)  # for reproducibility                                                          
# -- GPU setting
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu1,floatX=float32"

def run_train_RWC_dnn(options): 
    
    sr = options['sr']    
    num_frames = options['num_frames']
    max_bin = options['max_bin']
    min_bin = options['min_bin']
    note_res = options['note_res']
    drop_out = options['drop_out'] 
    optimizer = options['optimizer']
    activation = options['activation']
    output_activation = options['output_activation']
    select_ratio = options['select_ratio']     
    epoch = options['epoch']
   
    # -- DNN training
    batch_size = 128
    nb_epoch = epoch
    dim_input = num_frames*(max_bin-min_bin)
    dim_output = 40*note_res + 1 
    
    model = Sequential()
    #model.add(Dense(out_dim, input_dim, init))

    # -- hidden layer 1
    model.add(Dense(512, input_dim=dim_input, init='he_normal'))
    model.add(BatchNormalization(mode=0))
    model.add(Activation(activation)) #relu
    model.add(Dropout(drop_out))
    
    # -- hidden layer 2
    model.add(Dense(512, init='he_normal'))
    model.add(BatchNormalization(mode=0))
    model.add(Activation(activation))
    model.add(Dropout(drop_out))
    
    # -- hidden layer 3
    model.add(Dense(256, init='he_normal'))
    model.add(BatchNormalization(mode=0))
    model.add(Activation(activation))
    model.add(Dropout(drop_out))
    
    # -- output layer
    model.add(Dense(dim_output,init='he_normal'))
    model.add(BatchNormalization(mode=0))
    model.add(Activation(output_activation))

    
    if 'rms' == optimizer:
        rms = RMSprop()
        if output_activation == 'softmax':
            model.compile(loss='categorical_crossentropy', optimizer=rms,metrics=['accuracy'])   
        if output_activation == 'sigmoid':
            model.compile(loss='binary_crossentropy', optimizer=rms,metrics=['accuracy'])

#    if 'adadelta' == optimizer:      
#        adadelta = Adadelta();
#
#        if output_activation == 'softmax':
#            model.compile(loss='categorical_crossentropy', optimizer=adadelta,metrics=['accuracy'])   
#        if output_activation == 'sigmoid':
#            model.compile(loss='binary_crossentropy', optimizer=adadelta,metrics=['accuracy'])
#
#    if 'sgd' == optimizer:      
#        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#
#        if output_activation == 'softmax':
#            model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])   
#        if output_activation == 'sigmoid':
#            model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])

    # -- Save Weight file
    weight_file  =  os.getcwd()+'/model/'+'_frames_'+str(num_frames)+\
                    '_note_res_' + str(note_res)+'_drop_out_'+str(drop_out)+\
                    '_optimizer_'+ optimizer +'_activation_' + activation +\
                    '_output_activation_' + output_activation + '_TEST.hdf5'      

    if not os.path.exists(os.path.dirname(weight_file)):
        os.makedirs(os.path.dirname(weight_file))   
  
    if os.path.isfile(weight_file):
        print 'already trained!'
        model.load_weights(weight_file)
        print 'complete load weights!!'
    
    print 'weight_file : ', weight_file     
    print 'ready for training!\n'       
    # -- Save history of results  
    checkpointer = myModelCheckpoint(filepath=weight_file, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
  
    early_stopping = EarlyStopping(monitor='val_acc', patience=10)    
    
#    print '>>>>>>>>>>>>>>> training  >>>>>>>>>>>>>>>'
    for i  in range(25):
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> training : ' , str(i+1) ,'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        # -- load training data  
        x_train, y_train = load_train_data_RWC_dnn(dataset= 'train', shift='origin', sr = sr, num_frames=num_frames, min_bin=min_bin, max_bin=max_bin, note_res=note_res,voice_only = 1,select_ratio=select_ratio)
#        x_train_m1, y_train_m1 = load_train_data_RWC_dnn(dataset= 'train', shift='minus_1', sr = sr, num_frames=num_frames, min_bin=min_bin, max_bin=max_bin, note_res=note_res,voice_only = 1,select_ratio=select_ratio)
#        x_train_p1, y_train_p1 = load_train_data_RWC_dnn(dataset= 'train', shift='plus_1', sr = sr, num_frames=num_frames, min_bin=min_bin, max_bin=max_bin, note_res=note_res,voice_only = 1,select_ratio=select_ratio)

#        x_train = np.concatenate(( x_train_0,x_train_m1,x_train_p1), axis=0)
#        y_train = np.concatenate(( y_train_0,y_train_m1,y_train_p1), axis=0)    
       
#        rand_index = np.random.permutation(x_train.shape[0])
#        x_train = x_train[rand_index]
#        y_train = y_train[rand_index]

        # -- load validation data 
        x_valid, y_valid = load_train_data_RWC_dnn(dataset= 'valid', shift='origin', sr = sr, num_frames=num_frames, min_bin=min_bin, max_bin=max_bin, note_res=note_res,voice_only = 1,select_ratio=1.0)    
        
        # -- Standardization
        x_data_mean = x_train.mean(axis=0)
        x_data_std = x_train.std(axis=0)
        x_train = (x_train-x_data_mean)/x_data_std
        x_valid = (x_valid-x_data_mean)/x_data_std

        print ''
        print '================================='
        print 'x_train.shape : ' , x_train.shape 
        print 'y_train.shape : ' , y_train.shape 
        print 'x_valid.shape : ' , x_valid.shape 
        print 'y_valid.shape : ' , y_valid.shape
        print '================================='     
        print ''
        
        # -- Start Learning!!!
        model.fit(x_train, y_train, 
                  validation_data=(x_valid, y_valid), 
                  batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, 
                  callbacks=[checkpointer,early_stopping])     
       
   
    
if __name__ == '__main__':
    options_1= {
        'sr' : 8,
        'num_frames':11,        
        'min_bin': 0,
        'max_bin': 256,
        'note_res': 1,
        'drop_out' : 0.2,
        'optimizer' : 'rms',  # 'rms' , 'adadelta', 'sgd' 
        'activation' : 'relu', # 'relu'softplus
        'output_activation' : 'sigmoid', # 'softmax' sigmoid
        'select_ratio' : 0.05,
        'epoch' : 30
        
    }

    
    options_2 = {
        'sr' : 8,
        'num_frames':11,        
        'min_bin': 0,
        'max_bin': 256,
        'note_res': 2,
        'drop_out' : 0.2,
        'optimizer' : 'rms',  # 'rms' , 'adadelta', 'sgd' 
        'activation' : 'relu', # 'sigmoid'
        'output_activation' : 'sigmoid', # 'softmax' 'sigmoid'
        'select_ratio' : 0.1,
        'epoch' : 30
    }

    options_4 = {
        'sr' : 8,
        'num_frames':11,        
        'min_bin': 0,
        'max_bin': 256,
        'note_res': 4,
        'drop_out' : 0.2,
        'optimizer' : 'rms',  # 'rms' , 'adadelta', 'sgd' 
        'activation' : 'relu', # 'sigmoid'
        'output_activation' : 'sigmoid', # 'softmax' 'sigmoid'
        'select_ratio' : 0.1,
        'epoch' : 30
    }

         
    run_train_RWC_dnn(options_1)
#    run_train_RWC_dnn(options_2)
#    run_train_RWC_dnn(options_4)

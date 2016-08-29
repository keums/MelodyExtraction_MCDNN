# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:55:47 2016

@author: keums
"""
import numpy as np

def making_multi_frame(features, num_frames = 11): 
    '''
    making multi-frame spectrogram for training  

    |Parameters|
    ----------  
    features: shape(features)
        a specrogram of input .wav
    num_frame:
        the number of frame 
    
    |Returns|
    -------
    x: shape(num_feature[1],num_features[0]*num_frames) 
        the multi-frame featrues 
    '''    
    max_bin = 256
    min_bin = 0    
    max_num = np.shape(features)[1]
    x = np.zeros(shape = (max_num, num_frames*(max_bin-min_bin)))

    h_frames = int((num_frames-1)/2)
    total_num = 0

    for j in range(max_num):
        if num_frames > 1 :
            if j < h_frames:
                x[total_num] = np.reshape(features[min_bin:max_bin,0:num_frames], (num_frames*(max_bin-min_bin)))
            elif j >= max_num - h_frames:
                x[total_num] = np.reshape(features[min_bin:max_bin,np.shape(features)[1]- num_frames:], (num_frames*(max_bin-min_bin)))
            else:    
                x[total_num] = np.reshape(features[min_bin:max_bin,j-h_frames:j+h_frames+1], (num_frames*(max_bin-min_bin)))

        else:
            x[total_num] = features[min_bin:max_bin,j-h_frames:j+h_frames+1].T
        total_num = total_num + 1
    return x
    
    
    
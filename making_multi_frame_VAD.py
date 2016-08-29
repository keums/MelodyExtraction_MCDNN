# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 23:37:56 2016

@author: keums
"""
import numpy as np

def making_multi_frame_VAD(features, num_frames = 1, max_bin = 41, min_bin = 0): 
    '''
    making multi-frame features for training  

    |Parameters|
    ----------  
    features: shape(features)
        the result of SCDNN(res=1) of melody extraction.
    num_frame:
        the number of frame 
    max_bin:
        the size of features = 40 * note_res +1
    nim_bin: 0(default)
    |Returns|
    -------
    x: shape(num_feature[0], num_features[1]*num_frames) 
        the multi-frame featrues 
    '''    
    max_num = np.shape(features)[0]
    x = np.zeros(shape = (max_num, num_frames*(max_bin-min_bin)))

    h_frames = int((num_frames-1)/2)
    total_num = 0

    for j in range(max_num):
        if num_frames > 1 :
         
            if j < h_frames:
                x[total_num] = np.reshape(features[0:num_frames,min_bin:max_bin], (num_frames*(max_bin-min_bin)))
            elif j >= max_num-h_frames:
                x[total_num] = np.reshape(features[np.shape(features)[0]- num_frames:,min_bin:max_bin], (num_frames*(max_bin-min_bin)))
            else:    
                x[total_num] = np.reshape(features[j-h_frames:j+h_frames+1,min_bin:max_bin], (num_frames*(max_bin-min_bin)))

        else:
            x[total_num] = features[j:j+1,min_bin:max_bin]
                 
        total_num = total_num + 1
    x = x[:total_num]
   
    return x
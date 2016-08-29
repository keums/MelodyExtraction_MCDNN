# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:09:56 2016

@author: keums
"""

def mySelect_weight(note_res):

    weight_file = './model/_frames_11_note_res_' + str(note_res)+'_drop_out_0.2'+\
                    '_optimizer_rms_activation_relu_output_activation_sigmoid_Melody.hdf5' 

        
    return weight_file
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:24:18 2016

@author: keums
"""

import sys
import os
import numpy as np

from myFeatureExtraction import *
from viterbi import *

from making_multi_frame import *
from MelodyExtraction_SCDNN import *
from VAD_DNN import *

#plt.switch_backend('Qt4Agg')
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
np.random.seed(1337)  # for reproducibility                                                          
    
min_pitch = 38  # in MIDI                                                                                                                 
max_pitch = 78  # in MIDI    

def main(param = 0.2 , PATH_LOAD_FILE='/home/keums/Melody/dataset/adc2004_full_set/file/pop4.wav', PATH_SAVE_FILE='./SAVE_RESULTS/pop4.txt'):
   
#    PATH_LOAD_FILE = sys.argv[1]   
#    PATH_SAVE_FILE = sys.argv[2]   

    #==================================
    # Feature Extraction 
    # .wav --> spectrogram
    #==================================
    x_test_log = myFeatureExtraction(PATH_LOAD_FILE) #path ??

    #==================================
    # making multi column spectrogram 
    # for trainging 
    #==================================
    x_test_SF = making_multi_frame(x_test_log, num_frames = 1)
    x_test_MF = making_multi_frame(x_test_log, num_frames = 11)

    select_res_1st = 1
    select_res_2nd = 2
    select_res_3rd = 4
    pitch_range = np.arange(min_pitch, max_pitch + 1.0/select_res_3rd, 1.0/select_res_3rd)     
    
    #==================================
    # Melody extraction
    # using DNN 
    #==================================
    y_predict_1st = MelodyExtraction_SCDNN(x_test_MF, select_res_1st)
    y_predict_2nd = MelodyExtraction_SCDNN(x_test_MF, select_res_2nd)
    y_predict_3rd = MelodyExtraction_SCDNN(x_test_MF, select_res_3rd)
     
      
    #==================================
    # merge SCDNN
    #==================================
#    print 'Merging....'
    ratio_res_1_3 = select_res_3rd/select_res_1st
    ratio_res_2_3 = select_res_3rd/select_res_2nd

    y_predict_tmp_1_3 = np.zeros(y_predict_3rd.shape)
    y_predict_tmp_2_3 = np.zeros(y_predict_3rd.shape)
    
    for i in range(y_predict_3rd.shape[0]):
        for j in range(y_predict_1st.shape[1]-1):
            y_predict_tmp_1_3[i,j*ratio_res_1_3:j*ratio_res_1_3+ratio_res_1_3] = y_predict_1st[i,j]
        y_predict_tmp_1_3[i,-1] = y_predict_1st[i,-1]      

    for i in range(y_predict_3rd.shape[0]):
        for j in range(y_predict_2nd.shape[1]-1):
            y_predict_tmp_2_3[i,j*ratio_res_2_3:j*ratio_res_2_3+ratio_res_2_3] = y_predict_2nd[i,j]
        y_predict_tmp_2_3[i,-1] = y_predict_2nd[i,-1]      
    
#    y_predict = (y_predict_tmp_1_3+0.0000001) *(y_predict_tmp_2_3+0.0000001) * (y_predict_3rd +0.0000001)
    y_predict = 10**(np.log10(y_predict_tmp_1_3) +np.log10(y_predict_tmp_2_3)+ np.log10(y_predict_3rd))
    del y_predict_tmp_1_3
    del y_predict_tmp_2_3

    #==================================
    # singing voice detection
    #==================================
    voice_frame_vad= VAD_DNN(x_test_SF, y_predict_1st, param=0.2)

    #==================================
    # viterbi algorithm
    #==================================
    path_viterbi = './viterbi/'
    path_prior_matrix_file = path_viterbi + 'prior_'+ str(select_res_3rd) +'.npy'
    path_transition_matrix_file = path_viterbi + 'transition_matrix_'+ str(select_res_3rd)+ '.npy'    
    
    prior = np.load(path_prior_matrix_file)    
    transition_matrix = np.load(path_transition_matrix_file)   
    viterbi_path = viterbi(y_predict, transition_matrix=transition_matrix, prior=prior, penalty=0, scaled=True)    

    pitch_MIDI = np.zeros([y_predict.shape[0],1]) 
    pitch_freq = np.zeros([y_predict.shape[0],1])

    for i in range (y_predict.shape[0]):
        # for test : origianl 
#        index_predict[i] = np.argmax(y_predict[i,:])        
#        pitch_MIDI[i] = pitch_range[index_predict[i]]
        
        #viterbi_path
        pitch_MIDI[i] = pitch_range[viterbi_path[i]]
        pitch_freq[i] = 2**((pitch_MIDI[i]-69)/12.) * 440

    est_pitch = np.multiply(pitch_freq,voice_frame_vad)
    #==================================
    #adjust frame
    #==================================
 
    idx_shift = 2
    shift_array = np.zeros(idx_shift)            
    est_pitch = np.append(shift_array,est_pitch[:-idx_shift])

    #==================================
    # save result
    #==================================
       
    PATH_est_pitch = PATH_SAVE_FILE 

    if not os.path.exists(os.path.dirname(PATH_est_pitch)):
        os.makedirs(os.path.dirname(PATH_est_pitch))
    f= open(PATH_est_pitch,'w')

    for j in range(len(est_pitch)):
        est = "%f\t%f\n" % (0.01*j ,est_pitch[j])
        f.write(est)    
    f.close()  
    print PATH_est_pitch

if __name__ == '__main__':
    param = sys.argv[1]
    PATH_LOAD_FILE = sys.argv[2]
    PATH_SAVE_FILE = sys.argv[3]
 
    main(param, PATH_LOAD_FILE, PATH_SAVE_FILE)

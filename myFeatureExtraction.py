# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:10:41 2016

@author: keums
"""

import numpy as np
import librosa

def myFeatureExtraction(PATH_TEST_FILE):
    '''
    Compute a stft from a wav, and convert to log-scale the amplitude of a spectrogram

    |Parameters|
    ----------  
    PATH_TEST_FILE:
        path of .wav file
    
    |Returns|
    -------
    log_S: 
        log-scale the amplitude of a spectrogram
    '''
   
    y, sr = librosa.load(PATH_TEST_FILE, sr=8000)
    print PATH_TEST_FILE, sr, len(y)

    S = librosa.core.spectrum.stft(y, n_fft=1024, hop_length=80, win_length=1024)
    x_spec = np.abs(S)
    log_S = librosa.logamplitude(x_spec, ref_power=np.max)
    log_S = log_S.astype(np.float32)
    
    return log_S
        
if __name__ == '__main__':
    myFeatureExtraction() 
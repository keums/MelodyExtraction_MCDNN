# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 02:04:21 2016

@author: keums
"""
import os 
import numpy as np
# PATH of Training DATA 


PATH = os.getcwd()+'/Data/'
#data_path = '/media/bach1/dataset/RWC/pop/'
MIN_PITCH = 38  # in MIDI
MAX_PITCH = 78  # in MIDI

def load_train_data_RWC_dnn(dataset= 'train',shift='origin', sr=8,num_frames=11, min_bin=0, max_bin=256, note_res=1, voice_only=1 , select_ratio = 0.01):
    
    max_num = 1000000 #1500000 
    x = np.zeros(shape = (max_num, num_frames*(max_bin-min_bin)),dtype=np.float32)
    y = np.zeros(shape = (max_num))
    pitch_range = np.arange(MIN_PITCH, MAX_PITCH + 1.0/note_res, 1.0/note_res)

    # data 
    if 'train' == dataset:
        f = open(PATH+'filelist_train_label.txt')
    if 'valid' == dataset:
        f = open(PATH+'filelist_valid_label.txt')

    total_num = 0
    h_frames = int((num_frames-1)/2)

    i = 0

    for file_name in f:
        file_name = file_name.rstrip('\n')
        features = np.load(PATH + 'features/'+file_name.replace('.TXT','.npy'))
        print file_name
        pitch = np.load(PATH + 'pitch/'+ file_name.replace('.TXT','.npy'))

        # convert pitch to MIDI-like scale 
        labels = np.zeros(pitch.shape)
        voiced_onsets = np.zeros(pitch.shape)
        num_voice_onsets = 0

        for j in range(len(pitch)):
            if pitch[j] > 0.0 :
                if shift == 'minus_1':
                    labels[j] = (np.round(note_res*(69.0 + 12.0*np.log2(pitch[j]/440.0)))/float(note_res)) -1
                if shift == 'plus_1':
                    labels[j] = (np.round(note_res*(69.0 + 12.0*np.log2(pitch[j]/440.0)))/float(note_res)) +1
                elif shift == 'origin': 
                    labels[j] = np.round(note_res*(69.0 + 12.0*np.log2(pitch[j]/440.0)))/float(note_res)    
            # voice onset detection
            if j == 0:
                if pitch[j] > 0.0:
                    voiced_onsets[num_voice_onsets] = j
                    num_voice_onsets = num_voice_onsets+ 1
            else:
                if (pitch[j-1] == 0.0) and (pitch[j] > 0.0):
                    voiced_onsets[num_voice_onsets] = j
                    num_voice_onsets = num_voice_onsets + 1
        
        voiced_onsets = voiced_onsets[:num_voice_onsets]
        
        rand_index = np.random.choice(len(voiced_onsets)-num_frames,\
                                      int((len(voiced_onsets)-num_frames)))       
        count_notselect = 0
        if voice_only == 1:
            for jj in voiced_onsets[rand_index]:
                # iterate in an voiced segment
                j = int(jj)+h_frames
                while True:
                    num_voiced_frames = np.sum(labels[j-h_frames:j+h_frames+1]>0)
                    if  num_voiced_frames == num_frames:
                        if np.sum(labels[j-h_frames:j+h_frames+1]> MAX_PITCH) or np.sum(labels[j-h_frames:j+h_frames+1]<MIN_PITCH): 
#                            print('out of range')
                            count_notselect += 1
                        else:
                            if num_frames > 1 :
                                x[total_num] = np.reshape(features[min_bin:max_bin,j-h_frames:j+h_frames+1], (num_frames*(max_bin-min_bin)))
                            else:
                                x[total_num] = features[min_bin:max_bin,j].T
                            
                            y[total_num] = labels[j]
                            total_num = total_num + 1
                        j = j + 1
                        
                    else:
                        break
    
    f.close()

    x = x[:total_num]
    y = y[:total_num]
    
    # convert to one-hot representation
    y_labels = np.zeros(shape=(total_num, len(pitch_range)))
    for i in range(total_num):
        v = np.where(pitch_range == y[i])
        y_labels[i,v[0][0]] = 1
        
    # suffule
    if dataset == 'train':    
        rand_index  = np.random.choice(x.shape[0] ,int(x.shape[0]*select_ratio),replace = False)
        x = x[rand_index]
        y_labels = y_labels[rand_index]
    
    print '==' , dataset , ': ' , x.shape, y_labels.shape

    return x, y_labels

if __name__ == '__main__': 
    x, y_labels = load_train_data_RWC_dnn(dataset= 'train',note_res=1,shift='origin')
      

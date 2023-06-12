import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

def Masking(sig,threshold,n_fft=512,win_length=400,hop_length=160,eps=0.00000001):
     sig=sig/max(abs(sig)) #Make the signal to be within -1 to 1
     spectrogram = librosa.stft(np.copy(sig),n_fft=n_fft,hop_length=hop_length,win_length=win_length)
     magnitude, phase = librosa.magphase(spectrogram)
     
     #Estimating mask
     magnitude_db = 20*np.log10(magnitude+eps)
     magnitude_db_mask = 1*(magnitude_db > threshold)
     magnitude_mod = magnitude*magnitude_db_mask
     
     #Synthesizing the wavefile
     sig_mod=librosa.istft(magnitude_mod*phase,hop_length=hop_length,win_length=win_length)
     return sig_mod/max(abs(sig_mod)),magnitude_mod
     
baseDir = "masking/"
file_names = ["enroll_final1","test_nontarget_final1","test_target_final1","f11_1","f12_1","m11_1","m12_1"]   
threshold_list = [-20,-10,-5,0]
for i in range(0,len(file_names),1):
    name = file_names[i]  
    sig,sr = sf.read(baseDir + name + ".wav")
    for threshold in threshold_list:
        sig_est,magnitude_mod = Masking(sig,threshold)
        sf.write(baseDir + name + "_" + str(threshold) + ".wav",sig_est,sr)


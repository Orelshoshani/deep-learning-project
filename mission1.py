import pyrirgen
import numpy as np
import scipy as sp
import sounddevice as sd
import numpy.random as nprnd # white noise
import matplotlib.pyplot as plt
#import soundfile as sf
#import matplotlib.pyplot as plt
#from random import randrange
from scipy.io import wavfile
import scipy.signal

#sd.play(data, fs)
#status = sd.wait()  # Wait until file is done playing 

data_fs, data = wavfile.read('doors.wav')
sd.play(data, data_fs)
teta = np.arange(-45*np.pi/180, 136*np.pi/180, 5*np.pi/180)
R = 1
leng = len(teta)                                     #radius of speaker in [m]
s = np.zeros((leng, 3))
for i in np.arange(leng):
    s[i] = [3-R*np.cos(teta[i]), 3+R*np.sin(teta[i]), 1.5]  # Source position [x y z] (m)
print('finish1')
c = 342                                                     # Sound velocity (m/s)
fs = data_fs                                                # Sample frequency (samples/s)
r = [[3-0.05, 3-0.05, 1.5], [3+0.05, 3+0.05, 1.5]]          # Receiver position [x y z] (m)                                 
L = [6.2, 6, 2.5]                                           # Room dimensions [x y z] (m)
rt = 0.4                                                    # Reverberation time (s)
n = 1024*2                                                  # Number of samples
mtype = 'omnidirectional'                                   # Type of microphone
order = -1                                                  # Reflection order
dim = 3                                                     # Room dimension
orientation = 0                                             # Microphone orientation (rad)
hp_filter = True                                            # Enable high-pass filter

h_left_mic = np.zeros((leng, n))
h_right_mic = np.zeros((leng, n))
for i in np.arange(leng):    
    h_left_mic[i], h_right_mic[i] = pyrirgen.generateRir(L, s[i], r, soundVelocity=c, fs=fs, reverbTime=rt, nSamples=n, micType=mtype, nOrder=order, nDim=dim, isHighPassFilter=hp_filter)
print('finish2')

#conv of the 37 h_left and 37 h_right
data_len = len(data)

mic_left_in = np.zeros((leng, n+data_len-1))
mic_right_in = np.zeros((leng, n+data_len-1))

for i in np.arange(leng):
    mic_right_in[i] = sp.convolve(data, h_right_mic[i], mode='full')
    mic_left_in[i] = sp.convolve(data, h_left_mic[i], mode='full')
print('finish3')
# we have the results on the left&right mics of the 37 experiments

#create the noise signal
mean = 0
std = 0.01 
n = 1024*2 
num_samples = n+data_len-1
noise = nprnd.normal(mean, std, size=num_samples)
plt.plot(noise)
                              
#stft of the convolution result :
Zxx_R, Zxx_L = [], []
f_stft, t_stft, _ = sp.signal.stft(mic_right_in[0], fs=8000, window='hamming', nperseg=256, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)
for i in np.arange(leng):   
    _, _, Zxx_R1 = sp.signal.stft(mic_right_in[i], fs=8000, window='hamming', nperseg=256, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)
    Zxx_R.append(Zxx_R1)
    _, _, Zxx_L1 = sp.signal.stft(mic_left_in[i], fs=8000, window='hamming', nperseg=256, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)                                                                               #now we play the k experiment to the output
    Zxx_L.append(Zxx_L1)

print('finish STFT')
plt.pcolormesh(t_stft, f_stft, np.abs(Zxx_R[15])) # 15 for example


#play results 
k1, k2 = 36, 0
data_stereo = np.column_stack([mic_left_in[k1], mic_right_in[k1]])
sd.play(data_stereo, data_fs)
sd.wait()
data_stereo = np.column_stack([mic_left_in[k2], mic_right_in[k2]])
sd.play(data_stereo, data_fs)
sd.wait() 


#STFT, RTF 




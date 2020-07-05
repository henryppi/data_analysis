import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
import zipfile
import os, sys
import os.path
"""
adopted from
https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html
"""

filename = 'Un_bel_di_vedremo'
ext = 'm4a'

if not os.path.isfile(filename+'.wav'):
    # converting wav file from m4a file using specific time
    os.system('ffmpeg -i '+filename+'.'+ext+' -ss 00:00:00 -t 00:00:39 '+filename+'.wav')

# read wav file into array
fs, data = wavfile.read(''+filename+'.wav')

dt = 1./fs
samples = data.shape[0]
dur = samples/fs
print('samples',samples)
print('sample rate',fs)
print('time step',dt)
print('duration',dur)


fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(dt*np.arange(samples),data,'-b',lw=1)
ax.set_xlim([0, dt*samples])
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('sound time domain')
plt.savefig('sound_time_domain.png',dpi=200,format='png')
# plt.show()
plt.close()


yf = fft(data)
tf = np.linspace(0.0, 1.0/(2.0*dt), samples//2)
fig, ax = plt.subplots(figsize=(16, 8))
ax.semilogy(tf, 2.0/samples * np.abs(yf[0:samples//2]),'-r',lw=1)
ax.set_xlim([0, fs/2])
plt.grid()
plt.xlabel('frequency [Hz]')
plt.ylabel('log10 amplitude')
plt.title('sound FFT')
plt.savefig('sound_fft.png',dpi=200,format='png')
# plt.show()
plt.close()


M = 1024
shift = 100
slices = []
loop = True
k = 0
iA = 0
iB = M
while loop:
    slices.append(data[iA:iB])
    iA+=shift
    iB+=shift
    if iB>samples:loop=False
slices = np.array(slices)

win = np.hanning(M+1)[:-1]
slices = slices*win
slices = slices.T
spectrum = np.fft.fft(slices,axis=0)[:M // 2+1:-1]
spectrum = np.abs(spectrum)

fig, ax = plt.subplots(figsize=(16, 8))

S = np.abs(spectrum)
S = 20 * np.log10(S / np.max(S))

ax.imshow(S, origin='lower', cmap='viridis',
          extent=(0, dur, 0, fs / 2 / 1000))
ax.axis('tight')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]');
plt.title('sound spectrum')
plt.savefig('sound_spectrum.png',dpi=200,format='png')
# plt.show()
plt.close()

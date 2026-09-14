import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# fn = '/Users/hpp/Desktop/09-04-26/MixPre-017.WAV'

# sample_rate, data = wavfile.read(fn)

# print(sample_rate)
# print(data.shape)


import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SpectrumApp(QMainWindow):
    def __init__(self, wav_path):
        super().__init__()
        self.setWindowTitle("WAV Frequency Spectrum")
        self.resize(800, 500)
        
        # Main widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Matplotlib figure canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Compute and plot spectrum
        self.plot_spectrum(wav_path)
        
    def plot_spectrum(self, wav_path):
        sample_rate, data = wavfile.read(wav_path)
        
        data = data[:, 3]
        data = data[2_000_000:3_000_000]

        if 1:
            frequencies, times, Sxx = spectrogram(data, fs=sample_rate, nperseg=4096, noverlap=2048)

            ax = self.figure.add_subplot(111)
            im = ax.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
            self.figure.colorbar(im,ax=ax,label='Intensity [dB]')
            self.figure.tight_layout()
            self.canvas.draw()
        if 0:    
            # Compute FFT
            n = len(data)
            yf = np.fft.rfft(data)
            xf = np.fft.rfftfreq(n, 1 / sample_rate)
            
            # Amplitude magnitude (normalized and in dB)
            magnitude = 20 * np.log10(np.abs(yf) + 1e-6)
            
            # Plotting
            ax = self.figure.add_subplot(111)
            ax.plot(xf, magnitude, color='teal')
            ax.set_title("Audio Frequency Spectrum")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude (dB)")
            ax.grid(True)
            self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    fn = '/Users/hpp/Desktop/09-04-26/MixPre-017.WAV'


    # Replace with your actual wav file path
    ex = SpectrumApp(fn)
    ex.show()
    sys.exit(app.exec_())
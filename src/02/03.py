import numpy as np
import sounddevice as sd

from src.lib import signal


a = signal.Sin(A=1, f=400, phi=0)
b = signal.Sin(A=1, f=800, phi=0)
c = signal.Sawtooth(A=1, f=240, phi=0)
d = signal.Square(A=1, f=300, phi=0)

duration_s = 2.0
samplerate = 44100
t_cont = np.linspace(0, duration_s, samplerate)

for sig in a, b, c, d:
    sd.play(sig(t_cont), samplerate)
    sd.wait()

# scipy.io.wavfile.write("smth.wav", samplerate, d(t_cont).astype(np.int16))

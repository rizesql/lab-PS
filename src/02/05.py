import numpy as np
import sounddevice as sd

from src.lib import signal


duration_s = 2.0
samplerate = 44100
t_cont = np.linspace(0, duration_s, samplerate)


x1 = signal.Sawtooth(A=1, f=240, phi=0)(t_cont)
x2 = signal.Sawtooth(A=1, f=360, phi=0)(t_cont)
xs = np.concat([x1, x2], axis=0)

sd.play(xs, samplerate)
sd.wait()

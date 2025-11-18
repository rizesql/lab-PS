import numpy as np

from src.lib import signal

n = 20
n_experiments = 100

t = np.linspace(0, 2 * np.pi, n, endpoint=False)
x = signal.Sin(A=1, f=12, phi=0)(t) + signal.Square(A=1, f=5, phi=0)(t)

for d in np.random.randint(0, n, size=n_experiments):
    y = np.roll(x, d)

    X = np.fft.fft(x)
    Y = np.fft.fft(y)

    d_recovered_conv = np.fft.ifft(X * Y).real.argmax()

    # added a small epsilon to prevent division by zero
    d_recovered_phase = np.fft.ifft(Y / (X + 1e-10)).real.argmax()

    print(f"d={d:2d}  =>  Convolution recovery: {d_recovered_conv:2d}, Phase recovery: {d_recovered_phase:2d}")

    assert d_recovered_conv == d
    assert d_recovered_phase == d

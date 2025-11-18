import numpy as np

N = 100
low, high = -10, 10

p = np.random.randint(low, high + 1, size=N)
q = np.random.randint(low, high + 1, size=N)

r_direct = np.convolve(p, q).real.round()

print(f"r(x) (direct): {r_direct}")

r_size = len(p) + len(q) - 1
P = np.fft.fft(p, n=r_size)
Q = np.fft.fft(q, n=r_size)

r_fft = np.fft.ifft(P * Q).real.round()

print(f"r(x) (FFT):    {r_fft}")

assert np.allclose(r_direct, r_fft)

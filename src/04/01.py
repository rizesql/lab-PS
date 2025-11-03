import numpy as np
import time
from src.lib import fourier, signal
import matplotlib.pyplot as plt


def sig(t):
    sig1 = signal.Sin(A=1, f=10, phi=np.pi / 2)
    sig2 = signal.Sin(A=3, f=13.5, phi=0)
    sig3 = signal.Sin(A=1, f=7, phi=3 * np.pi / 4)

    return sig1(t) + sig2(t) + sig3(t)


N_values = np.array((128, 256, 512, 1024, 2048, 4096, 8192))
benchmark = np.zeros(shape=(N_values.size, 3))
for idx, N in enumerate(N_values):
    print(f"bench N={N}")
    t_disc = np.arange(0, 1, 1 / N)
    samples = sig(t_disc)

    start = time.perf_counter_ns()
    dft = fourier.dft(samples)
    end = time.perf_counter_ns()
    benchmark[idx, 0] = end - start

    start = time.perf_counter_ns()
    fft = fourier.fft(samples)
    end = time.perf_counter_ns()
    benchmark[idx, 1] = end - start

    start = time.perf_counter_ns()
    np_ft = np.fft.fft(samples)
    end = time.perf_counter_ns()
    benchmark[idx, 2] = end - start

print("\nBenchmark Results (ns):\n", benchmark)

benchmark_us = benchmark / 1000.0

labels = ["DFT", "FFT", "NumPy FFT"]

plt.figure(figsize=(10, 6))

plt.plot(N_values, benchmark_us[:, 0], "o-", label=labels[0])
plt.plot(N_values, benchmark_us[:, 1], "s-", label=labels[1])
plt.plot(N_values, benchmark_us[:, 2], "^-", label=labels[2])

plt.xscale("log")
plt.xticks(ticks=N_values, labels=N_values)
plt.yscale("log")

plt.title("Fourier Transform Performance Benchmark")
plt.xlabel("Number of Samples ($N$) (Log Scale)")
plt.ylabel("Execution Time ($\mu s$) (Log Scale)")

plt.grid()
plt.legend(title="Implementation")
plt.tight_layout()

plt.savefig("src/04/01.pdf", format="pdf")
plt.show()

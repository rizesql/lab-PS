import librosa
import numpy as np
import matplotlib.pyplot as plt

TARGET_SR = 22050
src = "src/04/vowels.wav"

sig, sr = librosa.load(src, sr=TARGET_SR, mono=True)

N = len(sig)
FRAME_SIZE = int(N * 0.01)
HOP_SIZE = max(1, int(FRAME_SIZE * 0.5))

spectrogram = []

for idx in range(0, N - FRAME_SIZE, HOP_SIZE):
    frame = sig[idx : idx + FRAME_SIZE]
    fft = np.fft.rfft(frame)

    spectrogram.append(np.abs(fft))

spectrogram = np.array(spectrogram).T
spectrogram_db = 20 * np.log10(spectrogram + 1e-9)
dbfs = spectrogram_db - np.max(spectrogram_db)

plt.imshow(
    dbfs,
    aspect="auto",
    origin="lower",
    cmap="inferno",
    extent=(0, N / TARGET_SR, 0, TARGET_SR / 2),
    vmax=-20,
    vmin=-100,
)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

cbar = plt.colorbar(format="%+2.0f")
cbar.set_label("dBFS")

plt.tight_layout()
plt.savefig("src/04/06.pdf", format="pdf")
plt.show()

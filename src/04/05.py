# Ok so nu m-am inregistrat pentru a obtine fisierul audio, in schimb am gasit
# [IPA vowel chart and audio](https://en.wikipedia.org/wiki/IPA_vowel_chart_with_audio)
# si am luat fisierele audio pentru principalele vocale. In acest fisier le concatenez
# intr-un singur wav pentru a il analiza in exercitiul urmator

import librosa
import numpy as np
import soundfile as sf

vowels = ["src/04/a.ogg", "src/04/e.ogg", "src/04/i.ogg", "src/04/o.ogg", "src/04/u.ogg"]

TARGET_SR = 22050

output = "src/04/vowels.wav"
res = []

for file in vowels:
    try:
        y, sr = librosa.load(file, sr=TARGET_SR, mono=True)

        res.append(y)

    except Exception as e:
        print(f"Error loading {file}: {e}")

try:
    sf.write(output, np.concatenate(res), TARGET_SR)
except Exception as e:
    print(f"Error saving file: {e}")

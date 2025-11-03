f_min = 40
f_max = 200

# Teorema Nyquist-Shannon stabileste ca frecventa minima de esantionare trebuie sa fie >
# decat limita superioara a intervalului in care este inclus un semnal band-pass.
nyquist_rate = 2 * f_max

epsilon = 10**-16
f_sample_min = nyquist_rate + epsilon

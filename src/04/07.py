p_sig = 90
snr_db = 80
snr = 10 ** (0.1 * snr_db)

p_noise = p_sig / snr
print(p_noise)

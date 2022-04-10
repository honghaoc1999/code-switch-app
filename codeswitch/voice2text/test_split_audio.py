#%%
import matplotlib.pyplot as plt
import numpy as np
import audiosegment

seg = audiosegment.from_file("audios/86696f34-b2c6-11ec-b44d-0a0791d41193.wav")
# Just take the first 3 seconds
hist_bins, hist_vals = seg[1:18000].fft()
hist_vals_real_normed = np.abs(hist_vals) / len(hist_vals)
plt.plot(hist_bins / 1000, hist_vals_real_normed)
plt.xlabel("kHz")
plt.ylabel("dB")
plt.savefig("test.png")
# %%

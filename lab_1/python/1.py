import matplotlib.pyplot as plt
import numpy as np

f = 12
T = int(1 / f)
A = 3
ph = np.pi / 8
L = 1/12  # длина волны

t = np.arange(0, L * 3, 1/(12*120))

y = 3 * np.sin(2*np.pi*f*t + ph)

plt.plot(t, y)
plt.xlabel('Time (t)')
plt.ylabel('Amplitude (V)')
plt.grid()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

f = 12
T = int(1 / f)
A = 3
ph = np.pi / 8
L = 1/12  # длина волны
#print(phase)

t = np.arange(0, L, 1/(12*120))
#print(x)

y = 3 * np.sin(2*np.pi*f*t + ph)
print(len(y))
print(len(t))

#w = 2 * f * 4 #частота дискретизации
#rray_w_y = []
#array_w_t = []
#for i in range(w):
#    div_w = i / w
 #   index = int(len(y) * div_w)
#    #print(index)
#    array_w_y.append(y[index])
#    array_w_t.append(t[index])


# FFT
yf = np.fft.fft(y)
xf = np.fft.fftfreq(1, 5)

plt.figure()
plt.plot(xf, 2.0 / L * np.abs(yf))
plt.plot(array_w_t, array_w_y)
plt.title('Спектр сигнала')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.grid()
plt.show()
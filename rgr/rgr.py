import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

# Transmitter
def ascii_coder(text):
    bit_sequence = []
    for symbol in text:
        ascii_symbol = ord(symbol)
        bits_symbol = bin(ascii_symbol)[2:].zfill(8)
        bit_sequence.extend(int(bit) for bit in bits_symbol)
    return bit_sequence

def computeCRC(packet, polynomial):
    packet_zeros = packet[:] + [0] * (len(polynomial) - 1)
    for i in range(len(packet)):
        if packet_zeros[i] == 1:
            for j in range(len(polynomial)):
                packet_zeros[i + j] ^= polynomial[j]
                
    CRC = packet_zeros[-(len(polynomial) - 1):]
    return CRC

def create_gold_sequence(x, y, len_sequence):
    gold_sequence = []
    
    for i in range(len_sequence):
        xor_shift_x = x[2] ^ x[4]
        xor_shift_y = y[2] ^ y[4]
        
        gold_sequence.append(x[-1] ^ y[-1])
        
        x.pop()
        y.pop()
    
        x.insert(0, xor_shift_x)
        y.insert(0, xor_shift_y)
        
    return gold_sequence

def bits_to_samples(bit_sequence, N):
    signal_samples = []
    for bit in bit_sequence:
        signal_samples.extend([bit] * N)
    return signal_samples


# Receiver
def NormalizedCorrelation(x, y):
    corr_array = []
    for i in range(len(x) - len(y) + 1):
        sumXY = 0.0
        sumX2 = 0.0
        sumY2 = 0.0
        corr = 0.0
        shifted_sequence = x[i:i + len(y)]
        for j in range(len(shifted_sequence)):
            sumXY += shifted_sequence[j] * y[j]
            sumX2 += shifted_sequence[j] * shifted_sequence[j]
            sumY2 += y[j] * y[j]
        corr = sumXY / np.sqrt(sumX2 * sumY2)
        corr_array.append(corr)
    return corr_array

def correlation_receiver(x, y, stop_word):
    corr_array = NormalizedCorrelation(x, y)
    start_useful_bits = np.argmax(corr_array)  
    print(f"Индекс начала полезного сигнала: {start_useful_bits}")
    corrected_signal = x[start_useful_bits:]
    
    corr_array_stop = NormalizedCorrelation(corrected_signal, stop_word)
    start_stop_word = np.argmax(corr_array_stop)
    corrected_signal = corrected_signal[:start_stop_word]
            
    return corrected_signal

def samples_to_bits(signal_samples, N):
    bit_sequence = []
    P = 0.5
    num_blocks = len(signal_samples) // N
    for i in range(num_blocks):
        block = signal_samples[i * N:(i + 1) * N]
        mean = np.mean(block)
        if mean >= P:
            bit_sequence.append(1)
        else:
            bit_sequence.append(0)
            
    return bit_sequence

def check_packet(received_packet, polynomial):
    result = computeCRC(received_packet, polynomial)
    
    return all(bit == 0 for bit in result)

def ascii_decoder(bit_sequence):
    text = ''
    for i in range(0, len(bit_sequence), 8):
        bits_symbol = bit_sequence[i:i + 8]
        if len(bits_symbol) < 8:
            break
        ascii_symbol = int(''.join(map(str, bits_symbol)), 2)
        text += chr(ascii_symbol)
    return text


def plot_spectrum(signal, title, fs):
    # N = len(signal)
    # T = N / fs
    # df = fs / N  # Разрешение по частоте

    # yf = fftpack.fft(signal)
    # xf = np.fft.fftfreq(N, 1/fs)[:N//2 + 1]  # +1 для чётных N

    # plt.plot(xf, 2/N * np.abs(yf[:N//2 + 1]), label=title)
    # plt.axvline(x=fs/2, color='r', linestyle='--', label='Частота Найквиста')
    N = len(signal)
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(N, 1/fs)[:N//2 + 1]  # Частоты от 0 до fs/2
    plt.plot(xf, 2/N * np.abs(yf[:N//2 + 1]), label=f'{title}, df={fs/N:.2f} Hz')
    plt.axvline(x=fs/2, color='r', linestyle='--')    



def my_plot_spectrum(signal, title, fs):
    N = len(signal)
    X = []
    for k in range(N):
        X.append(0)
        for n in range(N):
            X[k] += signal[n] * np.e ** (-np.j * 2 * np.pi * k * n / N)
            
    
    return X

    
    
    
    
def spectrum_plot(signal, fs, n, is_noisy):
    N = len(signal)
    spectrum = np.fft.fft(signal,n=N)
    freq = np.fft.fftfreq(N, 1/fs)
    amplitude = np.abs(spectrum)

    # print(amplitude)
    # Построение графика
    plt.figure(figsize=(12, 5))
    plt.stem(freq, amplitude,label='Амплитудный спектр')
    # plt.axvline(fs/2, color='r', linestyle='--', alpha=0.5, label='Частота Найквиста (25 Гц)')
    if is_noisy:
        plt.title(f'Амплитудный спектр зашумленного сигнала (fs={fs} Гц, N={n})')
    else:
        plt.title(f'Амплитудный спектр незашумленного сигнала (fs={fs} Гц, N={n})')        
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.xlim(-fs/2, fs/2)  # Показываем только полезный диапазон
    plt.grid()
    plt.legend()
    plt.show() 
    
    

# name = input("Enter name: ")
# surname = input("Enter surname: ")
name = "nik"
surname = "shap"
# 2

name_surname = name + " " + surname
bit_sequence = ascii_coder(name_surname)

plt.figure(figsize=(13, 10))
plt.step(range(len(bit_sequence)), bit_sequence, where='post', color='g', linewidth=2)
plt.yticks([0, 1], ['0', '1'])
plt.xlim(0)
plt.ylim(-0.5, 1.5)
plt.title('bit sequence, coding name and surname')

# 3
G = [1, 0, 0, 0, 0, 1, 1, 1]
CRC = computeCRC(bit_sequence, G)
CRC_print = ''.join(map(str, CRC))
print(f"CRC для битовой последовательности c данными: {CRC_print}")

bit_sequence_crc = bit_sequence + CRC

# 4
x = [1, 0, 1, 0, 1]
y = [1, 0, 1, 1, 1]
len_sequence = 31

gold_sequence = create_gold_sequence(x, y, len_sequence)
bit_sequence_crc_gold = gold_sequence + bit_sequence_crc

plt.figure(figsize=(13, 10))
plt.step(range(len(bit_sequence_crc_gold)), bit_sequence_crc_gold, where='post', color='g', linewidth=2)
plt.yticks([0, 1], ['0', '1'])
plt.xlim(0)
plt.ylim(-0.5, 1.5)
plt.title('Gold sequence + bit sequence + CRC')

# 4.1 Стоп слово для определения конца пакета
stop_word = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]
bit_sequence_crc_gold_stop = bit_sequence_crc_gold + stop_word

# 5
N = 10
signal_samples = bits_to_samples(bit_sequence_crc_gold_stop, N)

plt.figure(figsize=(13, 10))
plt.step(range(len(signal_samples)), signal_samples, where='post', color='g', linewidth=2)
plt.yticks([0, 1], ['0', '1'])
plt.xlim(0)
plt.ylim(-0.5, 1.5)
plt.title('Временные отсчёты сигнала для bit sequence')

# 6
signal = [0] * (2 * len(signal_samples))
# position = int(input(f"Введите номер позиции для вставки битовой последовательности (от 0 до {len(signal_samples)}): "))
position = 6
insert_length = min(len(signal_samples), (2 * len(signal_samples)) - position)
signal[position:position + insert_length] = signal_samples[:insert_length]

plt.figure(figsize=(13, 10))
plt.step(range(len(signal)), signal, where='post', color='g', linewidth=2)
plt.yticks([0, 1], ['0', '1'])
plt.xlim(0)
plt.ylim(-0.5, 1.5)
plt.title('Signal')

# 7
sigma = float(input("Введите значение отклонения (sigma): "))
# sigma = 0.001
noise = np.random.normal(0, sigma, 2 * len(signal_samples))
noisy_signal = [s + n for s, n in zip(signal, noise)]

plt.figure(figsize=(13, 10))
plt.step(range(len(noisy_signal)), noisy_signal, where='post', color='g', linewidth=2)
plt.yticks([0, 1], ['0', '1'])
plt.xlim(0)
plt.ylim(-1, 2)
plt.title('Зашумленный принятый сигнал')

# 8
corrected_signal = correlation_receiver(noisy_signal, bits_to_samples(gold_sequence, N), bits_to_samples(stop_word, N))

plt.figure(figsize=(13, 10))
plt.step(range(len(corrected_signal)), corrected_signal, where='post', color='g', linewidth=2)
plt.yticks([0, 1], ['0', '1'])
plt.xlim(0)
plt.ylim(-1, 2)
plt.title('Принятый сигнал без бит чистого шума')

# 9
bit_sequence_crc_gold_restored = samples_to_bits(corrected_signal, N)

plt.figure(figsize=(13, 10))
plt.step(range(len(bit_sequence_crc_gold_restored)), bit_sequence_crc_gold_restored, where='post', color='g', linewidth=2)
plt.yticks([0, 1], ['0', '1'])
plt.xlim(0)
plt.ylim(-1, 2)
plt.title('Восстановленные Gold sequence + bit sequence + CRC')

# 10 
bit_sequence_crc_restored = bit_sequence_crc_gold_restored[len_sequence:]

plt.figure(figsize=(13, 10))
plt.step(range(len(bit_sequence_crc_restored)), bit_sequence_crc_restored, where='post', color='g', linewidth=2)
plt.yticks([0, 1], ['0', '1'])
plt.xlim(0)
plt.ylim(-1, 2)
plt.title('Восстановленные bit sequence + CRC')

# 11
if check_packet(bit_sequence_crc_restored, G) == True:
    print("Ошибок в принятом пакете не обнаружено.")
# 12)
    bit_sequence_restored = bit_sequence_crc_restored[:-(len(G) - 1)]
    restored_text = ascii_decoder(bit_sequence_restored)
    print(f"Востановленный текст: {restored_text}")
else:
    print("Обнаружена ошибка в принятом пакете.")
    


# Параметры
N_array = [N//2, N, N*2]

fs = N  # Фиксированная частота дискретизации
# sigma = 2  # Уровень шума
position = 0  # Позиция вставки сигнала

plt.figure(figsize=(13, 10))
for N_i in N_array:
    
    signal_samples = bits_to_samples(bit_sequence_crc_gold_stop, N_i)
    # signal = np.zeros(2 * len(signal_samples))
    # signal[position:position+len(signal_samples)] = signal_samples
    
    # print("len(signal_samples): ", len(signal_samples))
    noise = np.random.normal(0, sigma, len(signal_samples))
    noisy_signal = signal_samples + noise
    
    # fs = fs_base * (N / N_ref)
    # plot_spectrum(noisy_signal, f'N={N}', fs)
    spectrum_plot(signal_samples, fs, N_i, 0)
    spectrum_plot(noisy_signal, fs, N_i, 1)
    # plot_spectrum_normalized(noisy_signal, f'N={N}')



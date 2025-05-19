import matplotlib.pyplot as plt
import numpy as np
# from scipy import fftpack
from scipy.signal import correlate
# from numba import njit


# Transmitter

def ascii_coder(text):
    bit_sequence = []
    for symbol in text:
        ascii_symbol = ord(symbol)
        bits_symbol = bin(ascii_symbol)[2:].zfill(8)
        bit_sequence.extend(int(bit) for bit in bits_symbol)
    return bit_sequence


def computeCRC(packet, polynomial):
    packet_zeros = packet.copy() + [0] * (len(polynomial) - 1)
    poly_len = len(polynomial)
    
    # Основной цикл вычисления CRC
    for i in range(len(packet)):
        if packet_zeros[i] == 1:
            # Применяем XOR с полиномом
            for j in range(poly_len):
                packet_zeros[i + j] ^= polynomial[j]
    
    # Возвращаем только CRC биты
    return packet_zeros[-(poly_len - 1):]

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
    """Четкое преобразование битов в отсчеты с проверкой"""
    if not all(bit in [0,1] for bit in bit_sequence):
        raise ValueError("Битовая последовательность должна содержать только 0 и 1")
    return [bit for bit in bit_sequence for _ in range(N)]





def fast_NormalizedCorrelation(x, y):
    """
    Векторизованная версия нормированной корреляции
    с использованием FFT для ускорения вычислений
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Кросс-корреляция через FFT
    corr = correlate(x, y, mode='valid', method='fft')
    
    # Нормировка
    x_norm = np.sqrt(np.cumsum(x**2)[len(y)-1:] - np.concatenate([[0], np.cumsum(x**2)[:-len(y)]]))
    y_norm = np.sqrt(np.sum(y**2))
    
    return corr / (x_norm * y_norm)

def fast_correlation_receiver(x, y, stop_word):
    """
    Ускоренная версия корреляционного приёмника
    x = signal
    y = bits_to_samples(gold_sequence, N)
    stop_word = bits_to_samples(stop_word, N)
    """
    # print("x_no_np: ", len(x))
    
    x = np.asarray(x)
    y = np.asarray(y)
    stop_word = np.asarray(stop_word)

    # Поиск начала сигнала
    corr = fast_NormalizedCorrelation(x, y)
    
    # Проверка на пустоту
    if corr.size == 0:
        return np.zeros(0)
        # raise ValueError("Кросс-корреляция пустая.")
    
    # Используем nanargmax для поиска максимума, игнорируя NaN
    start = np.nanargmax(corr)
    
    # Проверка на выход за пределы массива
    if start >= len(x):
        raise ValueError("start выходит за пределы длины сигнала x.")
    
    signal = x[start:]
    # print("len(signal): ", len(signal))
    # print("signal = x[start:] : ", signal.tolist())
    
    # Поиск конца сигнала
    stop_corr = fast_NormalizedCorrelation(signal, stop_word)
    if stop_corr.size == 0:
        # raise ValueError("Кросс-корреляция stop_corr пустая.")
        return np.zeros(5)
    # Проверяем, содержит ли stop_corr только NaN
    if np.all(np.isnan(stop_corr)):
        return np.zeros(100)
        # return signal[:len(signal) - len(stop)]
        # raise ValueError("Все значения в stop_corr равны NaN.")
    
    stop_pos = np.nanargmax(stop_corr)
    
    return signal[:stop_pos]















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
    # print(f"Индекс начала полезного сигнала: {start_useful_bits}")
    corrected_signal = x[start_useful_bits:]
    
    corr_array_stop = NormalizedCorrelation(corrected_signal, stop_word)
    start_stop_word = np.argmax(corr_array_stop)
    corrected_signal = corrected_signal[:start_stop_word]
            
    return corrected_signal


def samples_to_bits(signal_samples, N):
    bit_sequence = []
    P = 0.7
    num_blocks = len(signal_samples) // N
    for i in range(num_blocks):
        block = signal_samples[i * N:(i + 1) * N]
        mean = np.mean(block)
        if mean >= P:
            bit_sequence.append(1)
        else:
            bit_sequence.append(0)
            
    return bit_sequence

# @njit
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
    
    
# name = input("Enter name: ")
# surname = input("Enter surname: ")
name = "nik"
surname = "shap"
# 2

name_surname = name + " " + surname
bit_sequence = ascii_coder(name_surname)

print("bit_sequence( ", len(bit_sequence), "):", bit_sequence)

# 3
G = [1, 0, 0, 0, 0, 1, 1, 1]
CRC = computeCRC(bit_sequence, G)
CRC_print = ''.join(map(str, CRC))
# print(f"CRC для битовой последовательности c данными: {CRC_print}")

bit_sequence_crc = bit_sequence + CRC
# print("bit_sequence_crc( ", len(bit_sequence_crc), "):", bit_sequence_crc)


# 4
x = [1, 0, 1, 0, 1]
y = [1, 0, 1, 1, 1]
len_sequence = 31

gold_sequence = create_gold_sequence(x, y, len_sequence)
bit_sequence_crc_gold = gold_sequence + bit_sequence_crc

# 4.1 Стоп слово для определения конца пакета
stop_word = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]

bit_sequence_crc_gold_stop = bit_sequence_crc_gold + stop_word

# 5
N = 20
signal_samples = bits_to_samples(bit_sequence_crc_gold_stop, N)


# 6
signal = [0] * (2 * len(signal_samples))
# position = int(input(f"Введите номер позиции для вставки битовой последовательности (от 0 до {len(signal_samples)}): "))
position = 6
insert_length = min(len(signal_samples), (2 * len(signal_samples)) - position)
signal[position:position + insert_length] = signal_samples[:insert_length]

# print("signal_orig: ", signal)

# 7
# sigma = float(input("Введите значение отклонения (sigma): "))



sigma = np.arange(0, 3, 0.1)
# res = statistik_sigma(sigma, gold_sequence, N, 
#                      stop_word, signal, 
#                      len_sequence, G)


# print("sigma: ", sigma)




# Временная проверка шума
# test_noise = np.random.normal(0, 0.5, 1000)  # sigma=0.5
# plt.hist(test_noise, bins=50)
# plt.title("Распределение шума (sigma=0.5)")
# plt.show()

# test_signal = bits_to_samples(bit_sequence_crc_gold_stop, N)
# restored = fast_correlation_receiver(test_signal, 
#                               bits_to_samples(gold_sequence, N),
#                               bits_to_samples(stop_word, N))


# # print("Совпадение без шума:", restored == test_signal[6:-16*N])  # 6 - позиция вставки

# # plt.figure(figsize=(12,4))
# # plt.plot(test_signal, label='Оригинал')
# # plt.plot(restored, '--', label='Восстановлено')
# # plt.title("Сравнение сигналов без шума")
# # plt.legend()
# # plt.show()

# print(restored)
# print(bit_sequence)







noise = np.random.normal(0, 0.001, 2 * len(signal_samples))
# noisy_signal = [s + n for s, n in zip(signal, noise)]
noisy_signal = signal + noise
# print("noisy_signal:")
# print(noisy_signal.tolist())

corrected_signal = fast_correlation_receiver(signal, bits_to_samples(gold_sequence, N), bits_to_samples(stop_word, N))
print(corrected_signal)
# print("corrected_signal: len: :", len(corrected_signal), corrected_signal)


bit_sequence_crc_gold_restored = samples_to_bits(corrected_signal, N)
bit_sequence_crc_restored = bit_sequence_crc_gold_restored[len_sequence:]


print("bit_sequence_crc_restored: len: : ",  len(bit_sequence_crc_restored), bit_sequence_crc_restored)


print("bit_sequence_crc: len: : ",  len(bit_sequence_crc), bit_sequence_crc)

if bit_sequence_crc_restored != [] and  check_packet(bit_sequence_crc_restored, G) == True:
    print("Ошибок нет")
else:
    print("ошибка")






array_N = [10, 20, 40]
res_all = []
for N_i in array_N:
    N = N_i
    signal_samples = bits_to_samples(bit_sequence_crc_gold_stop, N)


    signal = [0] * (2 * len(signal_samples))
    position = 6
    insert_length = min(len(signal_samples), (2 * len(signal_samples)) - position)
    signal[position:position + insert_length] = signal_samples[:insert_length]

    res = np.zeros(len(sigma))
    for j in range(len(sigma)):
        
        cnt = 0
        max_i = 1000
        for i in range(max_i):
            
            if (sigma[j] == 0):
                noisy_signal = signal
            else:
                noise = np.random.normal(0, sigma[j], 2 * len(signal_samples))
                noisy_signal = signal + noise
            # noisy_signal = [s + n for s, n in zip(signal, noise)]
            
    
            corrected_signal = fast_correlation_receiver(noisy_signal, bits_to_samples(gold_sequence, N), bits_to_samples(stop_word, N))
            if (corrected_signal.any != -1 and corrected_signal.any != 0):
                bit_sequence_crc_gold_stop_restored = samples_to_bits(corrected_signal, N)
                bit_sequence_crc_restored = bit_sequence_crc_gold_stop_restored[len_sequence:]
            
            
                if bit_sequence_crc_restored != [] and check_packet(bit_sequence_crc_restored, G) == True:
                    cnt += 1
    
        print("j = ", j)
    
        res[j] = cnt / max_i
        
    res_all.append(res)





plt.figure(figsize=(12, 5))

plt.plot(sigma, res_all[0], label='N=10')
# plt.stem(sigma, res_all[0], "red")

plt.plot(sigma, res_all[1], label='N=20')
plt.plot(sigma, res_all[2], label='N=40')





plt.xlabel('sigma')
plt.ylabel('P декодирования')
# plt.xlim(-fs/2, fs/2)  # Показываем только полезный диапазон
plt.grid()
plt.legend()
plt.show() 















































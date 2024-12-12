import numpy as np
import matplotlib.pyplot as plt

MAPL_UL = 126.7
MAPL_DL = 144.1
d_km = np.linspace(0.01, 100, 1000)
d_m = d_km * 1000
hBS = 100
hms = 5
f_GHz = 1.8
f_MHz = f_GHz * 1000
radius_femto = 0.1
total_area = 100
business_area = 4

def UMiNLOS(d, f):
    return 26 * np.log10(f) + 22.7 + 36.7 * np.log10(d)

def COST231_urban(f, hBS, hms, d):
    A = 46.3
    B = 33.9
    a = 3.2 * (np.log10(11.75 * hms))**2 - 4.97
    s = np.where(d >= 1, 44.9 - 6.55 * np.log10(f), (47.88 + 13.9 * np.log10(f) - 13.9 * np.log10(hBS)) * (1 / np.log10(50)))
    Lclutter = 0
    return A + B * np.log10(f) - 13.82 * np.log10(hBS) - a + s * np.log10(d) + Lclutter

def COST231_DenseUrban(f, hBS, hms, d):
    A = 46.3
    B = 33.9
    a = 3.2 * (np.log10(11.75 * hms))**2 - 4.97
    s = np.where(d >= 1, 44.9 - 6.55 * np.log10(f), (47.88 + 13.9 * np.log10(f) - 13.9 * np.log10(hBS)) * (1 / np.log10(50)))    
    Lclutter = 3
    return A + B * np.log10(f) - 13.82 * np.log10(hBS) - a + s * np.log10(d) + Lclutter


PL_UMiNLOS = UMiNLOS(d_m, f_GHz)
PL_COST231 = COST231_urban(f_MHz, hBS, hms, d_km)
# PL_COST231 = COST231_DenseUrban(f_MHz, hBS, hms, d_km)

diff_COST231_UL = PL_COST231 - MAPL_UL
diff_COST231_DL = PL_COST231 - MAPL_DL
crossing_index_COST231_UL = np.where(np.diff(np.sign(diff_COST231_UL)))[0]
crossing_index_COST231_DL = np.where(np.diff(np.sign(diff_COST231_DL)))[0]
radius_COST231_UL = d_km[crossing_index_COST231_UL]
radius_COST231_DL = d_km[crossing_index_COST231_DL]
radius_COST231_arr = min(radius_COST231_UL, radius_COST231_DL)
radius_COST231 = radius_COST231_arr[0]

diff_UMiNLOS_UL = PL_UMiNLOS - MAPL_UL
diff_UMiNLOS_DL = PL_UMiNLOS - MAPL_DL
crossing_index_UMiNLOS_UL = np.where(np.diff(np.sign(diff_UMiNLOS_UL)))[0]
crossing_index_UMiNLOS_DL = np.where(np.diff(np.sign(diff_UMiNLOS_DL)))[0]
radius_UMiNLOS_UL = d_km[crossing_index_UMiNLOS_UL]
radius_UMiNLOS_DL = d_km[crossing_index_UMiNLOS_DL]
radius_UMiNLOS_arr = min(radius_UMiNLOS_UL, radius_UMiNLOS_DL)
radius_UMiNLOS = radius_UMiNLOS_arr[0]

area_macroBS = round(1.95 * (radius_COST231 ** 2), 2)
area_microBS = round(1.95 * (radius_UMiNLOS ** 2), 2)
area_femtoBS = round(1.95 * (radius_femto ** 2), 2)

number_macroBS_total = total_area / area_macroBS
number_macroBS_business = business_area / area_macroBS
number_microBS_business = business_area / area_microBS
number_femtoBS_business = business_area / area_femtoBS

number_macroBS_total = round(number_macroBS_total)
number_macroBS_business = round(number_macroBS_business)
number_microBS_business = round(number_microBS_business)
number_femtoBS_business = round(number_femtoBS_business)

print("Минимальный радиус макросоты (км):", round(radius_COST231, 2))
print("Площадь одной макросоты (кв. км):", area_macroBS)
print("Требуемое количество макросот (100 кв. км):", number_macroBS_total)
# print("Требуемое количество макросот (4 кв. км):", number_macroBS_business)
print("")
print("Минимальный радиус микросоты (км):", round(radius_UMiNLOS, 2))
print("Площадь одной микросоты (кв. км):", area_microBS)
print("Требуемое количество микросот (4 кв. км):", number_microBS_business)
print("")
print("Минимальный радиус фемтосоты (км):", round(radius_femto, 2))
print("Площадь одной фемтосоты (кв. км):", area_femtoBS)
print("Требуемое количество фемтосот (4 кв. км):", number_femtoBS_business)

plt.figure(figsize=(10, 6))
plt.plot(d_km, PL_UMiNLOS, label='UMiNLOS')
plt.plot(d_km, PL_COST231, label='COST231 (Город)')
# plt.plot(d_km, PL_COST231, label='COST231 (Плотная городская застройка)')
plt.axhline(y=MAPL_UL, color='green', linestyle='--', label='MAPL_UL')
plt.axhline(y=MAPL_DL, color='red', linestyle='--', label='MAPL_DL')

plt.title("Зависимость величины входных потерь радиосигнала от расстояния")
plt.xlabel('Расстояние между приемником и передатчиком, km')
plt.ylabel('Потери сигнала, dB')
plt.grid(True)
plt.legend()
plt.xlim(0, 20)
plt.ylim(0, 200)

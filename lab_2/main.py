import numpy as np
import matplotlib.pyplot as plt

d_km = np.linspace(0.01, 20, 1000)
d_m = d_km * 1000
f_GHz = 1.8
f_MHz = f_GHz * 1000
f = f_GHz * 1e9
hBS = 30
hms = 1
w = 25
b = 35

def FSPM(d, f):
    return 20 * np.log10((4 * np.pi * d * f) / 3e8)

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

class WalfishIkegami:
    def __init__(self, f, hBS, hms, w, b):
        self.f = f
        self.hBS = hBS
        self.hms = hms
        self.w = w
        self.b = b
        
    def Llos(self, d):
        return 42.6 + 20 * np.log10(self.f) + 26 * np.log10(d)
    
    def Lnlos(self, d):
        L0 = 32.44 + 20 * np.log10(self.f) + 20 * np.log10(d)
        L1 = self.L1_calc(d)
        L2 = self.L2_calc()
        
        if L1 + L2 > 0:
            return L0 + L1 + L2
        else:
            return L0
    
    def L1_calc(self, d):
        delta_h = self.hBS - self.hms
        
        if self.hBS > delta_h:
            L11 = -18 * np.log10(1 + self.hBS - delta_h)
        else:
            L11 = 0
        
        if self.hBS > delta_h:
            ka = 54  
        elif self.hBS <= delta_h and d > 0.5:
            54 - 0.8 * (self.hBS - delta_h)
        else:
            54 - 0.8 * (self.hBS - delta_h) * (d / 0.5)
            
        if self.hBS > delta_h:
            kd = 18
        else:
            kd = 18 - 15 * (self.hBS - delta_h) / delta_h
            
        kf = -4 + 0.7 * (self.f / 925 - 1)
        
        L1 = L11 + ka + kd * np.log10(d) + kf * np.log10(self.f) - 9 * np.log10(self.b)
        return L1
    
    def L2_calc(self):
        delta_h = self.hBS - self.hms
        phi = 30
        
        if 0 <= phi < 35:
            return -16.9 - 10 * np.log10(self.w) + 10 * np.log10(self.f) + 20 * np.log10(delta_h - self.hms) - 10 + 0.354 * phi
        elif 35 <= phi < 55:
            return -16.9 - 10 * np.log10(self.w) + 10 * np.log10(self.f) + 20 * np.log10(delta_h - self.hms) + 2.5 + 0.075 * phi
        else:
            return -16.9 - 10 * np.log10(self.w) + 10 * np.log10(self.f) + 20 * np.log10(delta_h - self.hms) + 4 - 0.114 * phi

    def lossesLlos(self, d):
        return self.Llos(d)
    
    def lossesLnlos(self, d):
        return self.Lnlos(d)

walfishIkegami = WalfishIkegami(f_MHz, hBS, hms, w, b)
PL_FSPM = FSPM(d_m, f)
PL_UMiNLOS = UMiNLOS(d_m, f_GHz)
PL_COST231_urban = COST231_urban(f_MHz, hBS, hms, d_km)
PL_COST231_DenseUrban = COST231_DenseUrban(f_MHz, hBS, hms, d_km)
PL_WalfishIkegami_Llos = [walfishIkegami.lossesLlos(d) for d in d_km]
PL_WalfishIkegami_Lnlos = [walfishIkegami.lossesLnlos(d) for d in d_km]

plt.figure(figsize=(10, 6))
plt.plot(d_km, PL_FSPM, label='FSPM')
plt.plot(d_km, PL_UMiNLOS, label='UMiNLOS', linestyle='--')
plt.plot(d_km, PL_COST231_urban, label='COST231 (Город)', linestyle=':')
plt.plot(d_km, PL_COST231_DenseUrban, label='COST231 (Плотная городская застройка)')
plt.plot(d_km, PL_WalfishIkegami_Llos, label='Walfish-Ikegami (Зона прямой видимости)')
plt.plot(d_km, PL_WalfishIkegami_Lnlos, label='Walfish-Ikegami (Зона не прямой видимости)', linestyle='-.')


plt.title("Зависимость величины входных потерь радиосигнала от расстояния")
plt.xlabel('Расстояние между приемником и передатчиком, km')
plt.ylabel('Потери сигнала, dB')
plt.grid(True)
plt.legend()
plt.xlim(0, 20)

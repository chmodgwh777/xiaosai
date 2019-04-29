import math
import numpy as np
from scipy.fftpack import fft,ifft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import platform

if platform.system() == 'Darwin':
    dataPath = '/Users/gao/Desktop/xiaosai/data'
else:
    dataPath = r'F:\大学工作\2018-2019 Sophomore year\190429第二次校赛建模\data.txt'

data = np.loadtxt(dataPath)
time = data[..., 0]

# extract data1 and compute vMean1 from data, 1 means the first driver
data1 = data[...,1]
accumulate1 = np.zeros(30)
s = 0
for i in range(30):
    s += data1[i]
    accumulate1[i] = s
vMean1 = accumulate1 / time
# extract data1 end

vMean1 = np.concatenate((vMean1[-1:], vMean1))
f = interp1d(np.linspace(0, 1, 31), vMean1, kind='cubic')
vInter = [f(t) for t in np.linspace(0, 1, 121)] # len(vInter) == 30k+1, where k is an integer

# fft
N = len(vInter)
vfft = fft(vInter)
vMod = abs(vfft)
vMod /= (N/2)
vMod[0] /= 2
varg = [math.atan(v.imag/v.real) for v in vfft]
vPositiveArg = [arg for arg in varg if arg <= 0.0][0:-1]

# this function is the result , defined of (0, 1)
def result(t, threshold = 0):
    s = 0.0
    for i in range(len(vPositiveArg)):
        if vMod[i] <= threshold:
            continue
        s += vMod[i]*math.cos(2*i*math.pi*t+vPositiveArg[i])
    return s

plot1 = 0
plot2 = 0
plot1 = 1
plot2 = 1
if plot1:
    t1 = np.linspace(0, 1, 31)
    plt.plot(t1, vMean1, 'r', label='rawData')
if plot2:
    t2 = np.linspace(0, 1, 1000)
    th1 = 0
    th2 = 1.04
    plt.plot(t2, [result(ti, th1) for ti in t2], 'b', label='threshold=%f'%th1)
    plt.plot(t2, [result(ti, th2) for ti in t2], 'g', label='threshold=%f'%th2)
plt.legend()
plt.show()
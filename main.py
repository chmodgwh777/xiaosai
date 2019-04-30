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

# extract data1 and compute vMean1 from data, 1 means the one of a driver
driver = 1
data1 = data[...,driver]
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
N = len(vInter) - 1
vfft = fft(vInter, N)
vMod = abs(vfft)
vMod /= (N/2)
vMod[0] /= 2
varg = [math.atan2(v.imag, v.real) for v in vfft][0:int(N/2)]
# vPositiveArg = [arg for arg in varg if arg <= 0.0][0:-1]

# this function is the result , defined of (0, 1)
def result(t, threshold = 0):
    s = vMod[0]
    for i in range(1, len(varg)):
        if vMod[i] <= threshold:
            continue
        s += vMod[i]*math.cos(2*i*math.pi*t+varg[i])
    return s

def printrRsult(t, threshold = 0):
    s = vMod[0]
    print(vMod[0], end='+')
    for i in range(1, len(varg)):
        if vMod[i] <= threshold:
            continue
        # s += vMod[i]*math.cos(2*i*math.pi*t+varg[i])
        print("%fCos(%dPit%+f)" % (vMod[i], 2*i, varg[i]), end='+')
    return s
# printrRsult(0, 0)

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
    th2 = 0.1
    plt.plot(t2, [result(ti, th1) for ti in t2], 'b', label='threshold=%f'%th1)
    plt.plot(t2, [result(ti, th2) for ti in t2], 'g', label='threshold=%f'%th2)
plt.title('Driver %d' % (driver))
plt.legend()
plt.show()
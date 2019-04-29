import math
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import platform

if platform.system() == 'Darwin':
    dataPath = '/Users/gao/Desktop/xiaosai/data'
else:
    dataPath = r'F:\大学工作\2018-2019 Sophomore year\190429第二次校赛建模\data'

N = 30

# init data, the type of data is numpy.ndarray
data = np.zeros((30, 4))
def readData(s):
    return np.array([float(f) for f in s.split()])
currentLine = ''
with open(dataPath, 'r') as f:
    currentLine = f.readline()
    i = 0
    while (currentLine):
        if currentLine[0].isdigit():
            data[i] = readData(currentLine)
            i += 1
        currentLine = f.readline()
time = data[..., 0]
# init data end




# extract data1 and compute vMean1 from data, 1 means the first driver
data1 = np.array([x[1] for x in data])
accumulate1 = np.zeros(30)
s = 0
for i in range(30):
    s += data1[i]
    accumulate1[i] = s
vMean1 = accumulate1 / time
# init data1 end


# fft
vfft = fft(vMean1)
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


plot1 = 1
plot2 = 1
if plot1:
    t1 = np.linspace(0, 1, 30)
    plt.plot(t1, vMean1)
if plot2:
    t2 = np.linspace(0, 1, 1000)
    plt.plot(t2, list(map(lambda t: result(t, 1.04), t2)))
    plt.plot(t2, list(map(lambda t: result(t, 0), t2)))
plt.show()
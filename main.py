import math
import numpy as np
from scipy.fftpack import fft,ifft
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import platform
from gao import *

if platform.system() == 'Darwin':
    dataPath = '/Users/gao/Desktop/xiaosai/data'
else:
    dataPath = r'F:\大学工作\2018-2019 Sophomore year\190429第二次校赛建模\data.txt'
driver = 1 # 司机1
data = np.loadtxt(dataPath) # 读取数据
time = data[..., 0] - 0.25
data1 = data[..., driver]
vlist = data1 * 2 # 求出平均速度

# 插值
F = interp1d(time, vlist, 'cubic', fill_value="extrapolate")
tInter = np.linspace(0, 15, 100) # 当前为100个插值点
vInter = [F(t) for t in tInter]


# 第一幅图
plt.subplot(211)
plt.plot(time, vlist, 'r.')
f = getFFTfun(vInter, 0, 15, 0.1)
draw(f, 0, 15, 1000).show()

# 第二幅图
plt.subplot(212)
I = [quad(f, v-0.5, v)[0] for v in data[..., 0]]
plt.plot(data[..., 0], data1, 'r.')
plt.plot(data[..., 0], I, 'g.')
plt.show()
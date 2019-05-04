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
driver = 3 # 司机1
data = np.loadtxt(dataPath) # 读取数据
time = data[..., 0] - 0.25
data1 = data[..., driver]
vlist = data1 * 2 # 求出平均速度

# 插值
F = interp1d(time, vlist, 'cubic', fill_value="extrapolate")
tInter = np.linspace(0, 15, 61) # 当前为61个插值点，包含原来的点
vInter = [F(t) for t in tInter]


# 第一幅图
N = 1000 # 绘制时的步长
threshold = 0.0
tPoint = np.linspace(0, 15, N)
plt.subplot(231)
plt.title('Result of Driver%d' % (driver))
plt.plot(np.linspace(0, 15, 15), np.ones(15)*100, 'y--') #超速的那条线
plt.plot(time, vlist, 'r.') #原始数据
obj = FFT(vInter, 0, 15, threshold) # obj为一对象，里面包含了模和辐角成员，该对象的getfun()方法反回得到的连续函数
f = obj.getfun() #用obj的getfun()方法得到fft后的连续函数
chaosu = [1 if f(t)>100 else 0 for t in tPoint]
print('超速小时:%.3fh' % (sum(chaosu)/N*15)) # 计算超速时间
draw(f, 0, 15, N).show() # draw为一封装好的对象，可以直接用show()方法来绘制f在[0, 15]上的图像

# 第二幅图，积分的比较
plt.subplot(232)
plt.title('Compare')
I = [quad(f, v-0.5, v)[0] for v in data[..., 0]]
plt.plot(data[..., 0], data1, 'r.')
plt.plot(data[..., 0], I, 'g.')

# 第三幅图，误差
plt.subplot(233)
plt.title('Error')
plt.plot(data[..., 0], I-data1, 'b.')

# 第四幅图，插值结果
plt.subplot(234)
plt.title('result of interpolate')
plt.plot(tInter, vInter, 'b.')
plt.plot(time, vlist, 'r.')
plt.plot(tInter, vInter, 'm:')

# 第五幅图，模
plt.subplot(235)
plt.title('mod')
plt.plot(tInter[:-1], obj.rawMod, 'b.')


plt.show()
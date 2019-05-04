import math
import numpy as np
from scipy.fftpack import fft,ifft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

print('Gao is loading...')

class draw:
    def __init__(self, fun, xmin, xmax, pointNum, color = 'b'):
        self.fun = fun
        self.xmin = xmin
        self.xmax = xmax
        self.pointNum = pointNum
        self.pointList = np.linspace(self.xmin, self.xmax, pointNum)
        self.valueList = np.array([fun(x) for x in self.pointList])
        self.color = color
    def show(self, c = 'b'):
        self.color = c
        plt.plot(self.pointList, self.valueList, self.color)

class FFT:
    def __init__(self, xlist, tmin, tmax, threshold = 0):
        self.xlist = xlist
        self.tmin = tmin
        self.tmax = tmax
        self.threshold = threshold
        self.N = len(self.xlist)-1
        self.xfft = fft(self.xlist, self.N)
        self.xMod = abs(self.xfft)
        self.rawMod = list(self.xMod)
        self.xMod /= (self.N/2)
        self.xMod[0] /= 2
        self.xarg = [math.atan2(x.imag, x.real) for x in self.xfft][0:int(self.N/2)]
    def getfun(self):
        def returnFun(t):
            tStd = (t-self.tmin)/(self.tmax-self.tmin)
            xmean = self.xMod[0]
            for i in range(1, len(self.xarg)):
                if self.xMod[i] <= self.threshold:
                    continue
                xmean += self.xMod[i]*math.cos(2*i*math.pi*tStd+self.xarg[i])
            return xmean
        return returnFun
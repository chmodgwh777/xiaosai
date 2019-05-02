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
    def show(self):
        plt.plot(self.pointList, self.valueList, self.color)

def getFFTfun(xlist, tmin, tmax, threshold = 0):
    N = len(xlist)-1
    xfft = fft(xlist, N)
    xMod = abs(xfft)
    xMod /= (N/2)
    xMod[0] /= 2
    xarg = [math.atan2(x.imag, x.real) for x in xfft][0:int(N/2)]
    def returnFun(t):
        tStd = (t-tmin)/(tmax-tmin)
        xmean = xMod[0]
        for i in range(1, len(xarg)):
            if xMod[i] <= threshold:
                continue
            xmean += xMod[i]*math.cos(2*i*math.pi*tStd+xarg[i])
        return xmean
    return returnFun

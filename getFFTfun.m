function fun = getFFTfun(xlist, tmin, tmax, threshold)
    N = length(xlist) - 1;
    xfft = fft(xlist, N);
    xMod = abs(xfft);
    xMod = xMod / (N/2);
    xMod(1) = xMod(1) / 2;
    xarg = arrayfun(@(x)atan2(imag(x), real(x)), xfft(1:floor(N/2)));
    function xmean = ffun(t)
        tStd = (t-tmin)/(tmax-tmin);
        xmean = xMod(1);
        for i = 2: length(xarg)
            if xMod(i) <= threshold
                continue;
            end
            xmean = xmean + xMod(i)*cos(2*(i-1)*pi*tStd+xarg(i));
        end
    end
    fun = @ffun;
end
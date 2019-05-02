function main()
    filename = '/Users/gao/Desktop/xiaosai/data';
    data = importdata(filename)';
    time = data(1, :);
    data1 = data(2, :);
    accumulate1 = zeros(1, 30);
    s = 0;
    for i = 1:30
        s = s + data1(i);
        accumulate1(i) = s;
    end
    vMean1 = accumulate1 ./ time;
    vMean1 = [vMean1(length(vMean1)), vMean1];

    % not interpolate yet
    vInter = vMean1;

    N = length(vInter) - 1;
    vfft = fft(vInter, N);
    vMod = abs(vfft);
    vMod = vMod / (N/2);
    vMod(1) = vMod(1) / 2;
    varg = zeros(1, floor(N/2));
    for i = 1:length(varg)
        varg(i) = atan2(imag(vfft(i)), real(vfft(i)));
    end

    function s = result(t, threshold)
        s = vMod(1);
        for i = 2:length(varg)
            if vMod(i) <= threshold
                continue
            end
            s = s + vMod(i)*cos(2*(i-1)*pi*t+varg(i));
        end
    end

    pointNum = 1000;
    t2 = linspace(0, 1, pointNum);

    result1 = zeros(1, pointNum);
    for k = 1:length(result1)
        result1(k) = result(t2(k), 0);
    end

    result2 = zeros(1, pointNum);
    for k = 1:length(result1)
        result2(k) = result(t2(k), 0.1);
    end

    plot(linspace(0, 1, 31), vMean1, t2, result1, t2, result2);

end


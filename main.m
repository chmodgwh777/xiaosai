N = 1000;
ddriver = 1;
data = importdata('data');
time = data(:, 1) - 0.25;
data1 = data(:, ddriver+1);
vlist = data1 .* 2;
vlist = vlist';
vtime = linspace(0, 15, 61);
interMethod = 'spline';
vInter = interp1(time, vlist, vtime, interMethod, 'extrap');

f = getFFTfun(vInter, 0, 15, 0.0);
x = linspace(0, 15, N);
y = arrayfun(f, x);

figure
s(1) = subplot(221);
plot(time, vlist, 'r.', x, y, 'b', 0:15, ones(1,16)*100, 'g');

s(2) = subplot(222);
I = arrayfun(@(v)quad(f, v-0.5, v), data(:, 1));
plot(data(:, 1), data1, 'r.');
hold on;
plot(data(:, 1), I, 'g.');

s(3) = subplot(223);
plot(data(:, 1), I-data1, 'b.');

s(4) = subplot(224);
plot(time, vlist, 'r+');
hold on;
plot(vtime, vInter, 'b.');

title(s(1), sprintf('result of driver%d', ddriver), 'FontSize', 15);
title(s(2), 'compare the integrate value', 'FontSize', 15);
title(s(3), 'Error of the integrate', 'FontSize', 15);
title(s(4), sprintf('result of interpolate, method:%s', interMethod), 'FontSize', 15);
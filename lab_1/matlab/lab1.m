f=12
% #y(t)=3*sin(2 * pi * f * t + pi / 8)
Ts = 1/(12*120)
t = 0:Ts:1-Ts;
x = 3 * sin(2 * pi *f*t + pi/8);
plot(t, x);
w=2*f 
y = fft(x);
fs = 1/Ts;
f = (0:length(y)-1)*fs/length(y);
% f = 0:5:20;
plot(f, abs(y))
% ylabel('Amplitude')
% xlabel('Time(s)')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
title('Magnitude')
clc;
close all;
clear;
f = fopen('/media/songlab2/easystore/field_test/v1green_surround_RX2.cf32', 'rb');
%f = fopen('data2M.c8', 'rb');

fs=25e6;
%fs = 2e6;
ts = 1./fs;

count = 2*fs; % take n seconds
%v = fread(f,count);

t = fread(f, [2, count], 'float');
%t = fread(f, [2, 50*fs], 'float');

fclose(f);

v = t(1,:) + t(2,:)*1i;
[r, c] = size (v);
v = reshape (v, c, r);

%figure
%subplot(2,1,1);
%plot(abs(v));
%subplot(2,1,2);
%plot(abs(fftshift(fft(v,1024))))
%figure
%spectrogram(v);

figure
resultMatrix0 = [];
cursor = 1;
windowLength = 1024;
for i = 1:1000
    fftVector = abs(fftshift(fft(v(cursor:cursor+windowLength),1024)));
    
    resultMatrix0 = [resultMatrix0;fftVector'];
    cursor = cursor + windowLength;
end
imagesc(resultMatrix0');

figure
subplot(2,1,1)
plot(atan2(imag(v(1:800)),real(v(1:800))),'*');
subplot(2,1,2)
plot(unwrap(atan2(imag(v(1:800)),real(v(1:800))),'*'));
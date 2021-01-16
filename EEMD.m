clc;

close all;

clear;

tic;

f = fopen('C:/Users/59699/Desktop/ERAU/RA/test/field_test/v1green_surround_RX2.cf32', 'rb');

fs=25e6;

ts = 1./fs;


count = 2*fs; % take n seconds


t = fread(f, [2, count], 'float');


fclose(f);


v = t(1,:) + t(2,:)*1i;

[r, c] = size (v);

v = reshape (v, c, r);


v = v(1:1000:end,:);


t = (0:length(v)-1)/fs;


IMF1=eemd2(v,0.1,20); %eemd

finalmatrix = [];

for i=2:1:6

    cursor = 1;

    windowLength = 256;

    v = IMF1(:,i);

    matrix = [];

for j = 1:floor(length(v)/windowLength)

    fftVector = abs(fftshift(fft(v(cursor:cursor+windowLength),256)));

   

    matrix = [matrix;fftVector'];

    cursor = cursor + windowLength;

end

    finalmatrix=[finalmatrix,matrix];

end

save('data.mat','finalmatrix')

toc;
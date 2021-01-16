clc;
close all;
clear;
f = fopen('G:\Test Data\sampling25(v1).cf32', 'rb');


fs=25e6;
ts = 1./fs;

count = 2*fs; % take n seconds


t = fread(f, [2, count], 'float');


fclose(f);

v = t(1,:) + t(2,:)*1i;
[r, c] = size (v);
v = reshape (v, c, r);
v = real(v);
v = v(1:1000:end,:);
%filename = 'test.mat';
%save(filename,'v');

%f = filename;

t = (0:length(v)-1)/fs;
[imf, residual, info] = emd(v,'Interpolation','pchip','Display',1);

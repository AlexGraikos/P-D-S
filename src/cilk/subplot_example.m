% A simple example of how to use subplot add multiple plots in one MATLAB
% figure
%
% date 23 Nov 2014

clear all
close all


x=0:.1:2*pi;
subplot(2,2,1);
plot(x,sin(x));


subplot(2,2,2);
plot(x,cos(x));


subplot(2,2,3)
plot(x,exp(-x));


subplot(2,2,4);
plot(peaks);


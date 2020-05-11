function [output] = downsample(input, time, sampled_length, interval_length)

%the function for downsampling the data with given length

rate = sampled_length / interval_length;
for_north = linspace(2,4*rate-2,rate);
for_south = for_north + 1;
for_all = for_south + 1;
a = size(input);
x = linspace(sampled_length,time,time/sampled_length);
%input = reshape(input,[a(3),a(4)]);
%a = size(input);
h = reshape(input.', [4*rate,(a(1)*a(2)/(4*rate))]).';  %4 = col number of y_eve and y_rx

output = [transpose(x) sum(h(:,for_north),2) sum(h(:,for_south),2) sum(h(:,for_all),2)];
end

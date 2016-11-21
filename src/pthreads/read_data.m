function [results] = read_data(src, nL, nP, nN,iter)
% function [results] = read_data(src, nL, nP, nN,iter)
% 
% input: 
%   src: The file containing the raw results, produced by the run_tests.sh script 
%   nL: The number of different tree heights tested
%   nP: The number of different population thresholds tested
%   nN: The number of different number of particles tested
%   iter: Number of repetitions of a single data point
%
% output:
%   results: The structure with the final results 
%       .hash: The time to compute the hash codes
%       .morton: The time to compute the morton encoding
%       .sort: The time of the truncated sort
%       .dataR: The time for data rearrangement
%
% author: Nikos Sismanis
% date: 22 Nov 2014

system(sprintf('grep "Time to compute the hash codes" %s > hash.txt', src));
system(sprintf('grep "Time to compute the morton encoding" %s > morton.txt', src));
system(sprintf('grep "Time for the truncated radix sort" %s > sort.txt', src));
system(sprintf('grep "Time to rearrange the particles in memory" %s > data_rear.txt', src));


fid = fopen('hash.txt');
hash_time = fscanf(fid, 'Time to compute the hash codes            : %fs\n');
fclose(fid);

hash_mean = mean(reshape(hash_time, iter, length(hash_time) / iter), 1);
hash_mean = reshape(hash_mean, nL, nP, nN);

fid = fopen('morton.txt');
morton_time = fscanf(fid, 'Time to compute the morton encoding       : %fs\n');
fclose(fid);

morton_mean = mean(reshape(morton_time, iter, length(morton_time) / iter), 1);
morton_mean = reshape(morton_mean, nL, nP, nN);

fid = fopen('sort.txt');
sort_time = fscanf(fid, 'Time for the truncated radix sort         : %fs\n');
fclose(fid);

sort_mean = mean(reshape(sort_time, iter, length(sort_time) / iter), 1);
sort_mean = reshape(sort_mean, nL, nP, nN);

fid = fopen('data_rear.txt');
dataR_time = fscanf(fid, 'Time to rearrange the particles in memory : %fs\n');
fclose(fid);

dataR_mean = mean(reshape(dataR_time, iter, length(dataR_time) / iter), 1);
dataR_mean = reshape(dataR_mean, nL, nP, nN);

results.hash = hash_mean;
results.morton = morton_mean;
results.sort = sort_mean;
results.dataR = dataR_mean;


end

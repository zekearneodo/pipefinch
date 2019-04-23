useGPU = ${use_gpu}; % do you have a GPU? Kilosorting 1000sec of 32chan simulated data takes 55 seconds on gtx 1080 + M2 SSD.

% add to path the kilo_dir and npymat_dir
addpath(genpath('${kilo_dir}'))
addpath(genpath('${npy_matdir}'))

% This part adds paths
disp(('${kilo_dir}'))
addpath(('${kilo_dir}')) % path to kilosort folder
addpath(('${npy_matdir}')) % path to npy-matlab scripts

pathToYourConfigFile = ('${data_dir}');

% Run the configuration file, it builds the structure of options (ops)
% So there shouldnt be ops in this file!!

run(fullfile(pathToYourConfigFile, 'config.m'))
% this one will populate ops structure.
disp('config.m ran; ops:')
disp(ops)

% preprocess data to create temp_wh.dat
rez = preprocessDataSub(ops);

% time-reordering as a function of drift
rez = clusterSingleBatches(rez);
rootZ = ops.root; 
save(fullfile(rootZ, 'rez.mat'), 'rez', '-v7.3');

% main tracking and template matching algorithm
rez = learnAndSolve8b(rez);

% final merges
rez = find_merges(rez, 1);

% final splits by SVD
rez = splitAllClusters(rez, 1);

% final splits by amplitudes
rez = splitAllClusters(rez, 0);

% decide on cutoff
rez = set_cutoff(rez);

fprintf('found %d good units \n', sum(rez.good>0))

% write to Phy
fprintf('Saving results to Phy  \n')
rezToPhy(rez, rootZ);

%% if you want to save the results to a Matlab file... 

% discard features in final rez file (too slow to save)
rez.cProj = [];
rez.cProjPC = [];

% save final results as rez2
fprintf('Saving final results in rez2  \n')
fname = fullfile(rootZ, 'rez2.mat');
save(fname, 'rez', '-v7.3');

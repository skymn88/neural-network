% Load your adapter, then compare each flight without amplitude normalization.
projectDir=fileparts(mfilename('fullpath'));addpath(projectDir,'-begin');setup_project();
% Scale synthetic units to YOUR units in order: current, DDV, RAM, angle.
% Example: RAM in mm -> [1 1 1000 1]. DDV requires your stroke calibration.
syntheticScale=[1 1 1 1];
% Signed scale / offset only for KNOWN sensor calibration, not visual fitting.
syntheticOffset=[0 0 0 0];
bb_plot_user_vs_synthetic(syntheticScale,syntheticOffset);

function cfg = bb_config()
% Repository configuration: compact reports, optional force, position-output-error FIR.
% MATLAB R2019a + Deep Learning Toolbox. No Signal Processing Toolbox needed.
cfg.version = 'plant';
cfg.diagnostics = false; % opt-in alignment, selected-stage and hold drift reports
cfg.alignmentMaxLag = 64;
cfg.printTestForAll = true;
cfg.Fs = 64;
cfg.simulinkOutputHz = 200;   % interface only; NN trains/advances at cfg.Fs
cfg.requireForce = false;  % force is not a target in the active four-signal chain
cfg.betaCandidates = [0 -.003 -.01 -.03 -.1 -.3 -1];
cfg.oeLags = [32 128 256];
cfg.oeRidge = [1e-6 1e-3 .1];
cfg.oeMaxTrainRows = 21000;
cfg.usePressure = false;    % true -> RAM NN also needs SupplyPressure, ReturnPressure
cfg.baselineOutputs = {'DDV','RAM','ActuatorForce','Deflection'};
cfg.inputLags = 32;        % 0.5 s lookback at 64 Hz
cfg.outputLags = 8;        % 0.125 s autoregressive lookback
cfg.hiddenSizes = [16 16];
cfg.componentMode = 'mlp';  % DDV/RAM component defaults to MLP; 'narx' optional
cfg.epochs = 150;          % smoke_test reduces this temporarily
cfg.maxTrainRows = 100000;
cfg.seed = 42;
cfg.maxGapFactor = 5;
cfg.guardSeconds = 1.0;   % buffer between 1-flight chronological splits
cfg.makePlots = true;
paths=bb_paths();cfg.outputRoot=paths.results;
if ~exist(cfg.outputRoot,'dir'),mkdir(cfg.outputRoot);end
cfg.runPrefix = 'results';
end

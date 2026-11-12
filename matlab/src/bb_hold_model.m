function m=bb_hold_model(cfg)
% Honest baseline: keep only initial RAM; one-step is previous measured RAM.
m.name='HoldInitial';m.kind='FIR';m.mode='mlp';m.Fs=cfg.Fs;
m.inputNames={'DDV'};m.outputNames={'RAMRate'};m.nInputs=1;m.nOutputs=1;
m.inputLags=0;m.outputLags=0;m.xMean=0;m.xStd=1;m.yMean=0;m.yStd=1;
m.weights=[0;0];m.beta=0;m.rateGain=1;m.rateBias=0;
end

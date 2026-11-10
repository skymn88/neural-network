function [c,kind]=bb_rich_config(cfg,label)
c=cfg;c.hiddenSizes=[24 16];c.maxTrainRows=12600;c.trainingSampling='mixed';
c.trainFcn='trainscg';c.epochs=400;c.gpMaxPoints=600;c.gpNoiseVariance=.002;c.gpCoverage=true;
kind='MLP';
switch label
    case 'NL_OE',kind='NL';
    case 'NL_FB',kind='FB';
    case {'MLP_basic','MLP_v6_recipe'} % historical config alias
        c.hiddenSizes=[16 16];c.maxTrainRows=8400;c.trainingSampling='uniform';c.epochs=150;
    case 'MLP_more_training'
    case 'MLP_quasiNewton'
        c.trainFcn='trainbfg';c.epochs=300; % MATLAB BFGS; Python uses L-BFGS.
    case 'FIR_rich',kind='FIR';
    case 'GP_coverage',kind='GP';
    otherwise,error('Unknown rich candidate');
end
if isfield(cfg,'quick')&&cfg.quick,c.epochs=5;c.maxTrainRows=700;c.gpMaxPoints=150;end
end

function [x,rate,info]=bb_rollout_rate(model,flight,x0,override)
% Learned rate + one internal position state. No measured outputs after x0.
if nargin<4,override=struct();end
if ~isscalar(x0)||~isfinite(x0),error('Initial RAM must be one finite physical value');end
a=1+model.beta/model.Fs;
if ~(a>0 && a<=1),error('Invalid integrator coefficient');end
rate=bb_predict_batch(model,flight,override);
if isfield(model,'rateGain'),rate=model.rateGain*rate;end
if isfield(model,'rateBias'),rate=rate+model.rateBias;end
if isfield(model,'wh'),rate=bb_wh_map(model.wh,rate);end
if isfield(model,'wh')&&~isfield(model,'ramBounds')&&~isfield(model,'stateLeak')
    x=bb_wh_rollout(model,rate,x0);info=struct('diverged',any(~isfinite(x)));
    if isfield(model,'divergenceThreshold'),info.diverged=info.diverged||any(abs(x)>model.divergenceThreshold);end
    return;
end
if size(rate,2)~=1&&~isfield(model,'dynamicMLP'),error('Rate model must have one output');end
info=struct('diverged',false);
if isfield(model,'divergenceThreshold'),info.diverged=abs(x0)>model.divergenceThreshold;end
if isfield(model,'constantRAM'),x0=model.constantRAM;end
x=zeros(size(rate,1),1);x(1)=x0;
if isfield(model,'ramBounds'),x(1)=min(max(x(1),model.ramBounds(1)),model.ramBounds(2));end
for k=2:numel(x)
    x(k)=bb_ram_step(model,x(k-1),rate(k,:));
    if isfield(model,'divergenceThreshold')
        info.diverged=info.diverged||~isfinite(x(k))||abs(x(k))>model.divergenceThreshold;
    end
    if isfield(model,'ramBounds'),x(k)=min(max(x(k),model.ramBounds(1)),model.ramBounds(2));end
end
if isfield(model,'dynamicMLP')
 info.externalInputs=rate;rate=[0;diff(x)]*model.Fs;
end
end

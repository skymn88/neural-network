function [x,rate]=bb_rollout_rate(model,flight,x0,override)
% Learned rate + one internal position state. No measured outputs after x0.
if nargin<4,override=struct();end
if ~isscalar(x0)||~isfinite(x0),error('Initial RAM must be one finite physical value');end
a=1+model.beta/model.Fs;
if ~(a>0 && a<=1),error('Invalid integrator coefficient');end
rate=bb_predict_batch(model,flight,override);
if isfield(model,'rateGain'),rate=model.rateGain*rate;end
if isfield(model,'rateBias'),rate=rate+model.rateBias;end
if size(rate,2)~=1,error('Rate model must have one output');end
x=zeros(size(rate));x(1)=x0;
for k=2:numel(x)
    x(k)=a*x(k-1)+rate(k)/model.Fs;
end
end

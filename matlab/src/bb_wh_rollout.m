function x=bb_wh_rollout(model,rate,x0)
% Vectorized one-state rollout. rate is AFTER h; first sample only sets x0.
a=1+model.beta/model.Fs;
if ~(a>0&&a<=1),error('Invalid Wiener integrator coefficient');end
rate=rate(:);drive=rate;drive(1)=0;k=(0:numel(rate)-1)';
x=x0*a.^k+filter(1/model.Fs,[1 -a],drive);
end

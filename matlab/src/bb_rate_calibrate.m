function [gain,bias]=bb_rate_calibrate(model,flights,start)
% Fit scalar rate gain/bias against whole TRAIN trajectories, without resets.
if nargin<3,start=66;end
A=[];Y=[];a=1+model.beta/model.Fs;
if ~(a>0&&a<=1),error('Invalid integrator');end
for f=1:numel(flights)
    fl=flights(f);g=bb_predict_batch(model,fl);k=(0:numel(g)-1)';
    response=[0;filter(1/model.Fs,[1 -a],g(2:end))];
    if model.beta==0,constant=k/model.Fs;else,constant=(1-a.^k)/(-model.beta);end
    initial=fl.S.RAM(1)*a.^k;
    ix=start:numel(g);
    A=[A;response(ix) constant(ix)];Y=[Y;fl.S.RAM(ix)-initial(ix)]; %#ok<AGROW>
end
if rank(A)<2
    warning('bb:calibrationRank','Skipping rate calibration: insufficient excitation');
    gain=1;bias=0;return;
end
w=A\Y;gain=w(1);bias=w(2);
end

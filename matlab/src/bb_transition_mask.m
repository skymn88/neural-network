function mask=bb_transition_mask(x,Fs,threshold,seconds)
% Causal diagnostic/weight mask: threshold event plus the following interval.
if threshold<0||seconds<0||Fs<=0,error('Invalid transition mask settings');end
x=x(:);event=[false;abs(diff(x))>threshold];
v=conv(double(event),ones(round(seconds*Fs)+1,1),'full');mask=v(1:numel(x))>0;
end

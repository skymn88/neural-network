function [slope,worst,count]=bb_hold_drift(error,mask,Fs)
% Separate contiguous holds; sample-weighted signed slope and worst magnitude.
error=error(:);mask=mask(:);edges=diff([false;mask;false]);a=find(edges==1);b=find(edges==-1)-1;
slope=NaN;worst=NaN;count=0;total=0;weight=0;
for j=1:numel(a)
 ix=a(j):b(j);if numel(ix)<2||any(~isfinite(error(ix))),continue;end
 t=(0:numel(ix)-1)'/Fs;t=t-mean(t);v=t'*error(ix)/(t'*t);
 total=total+numel(ix)*v;weight=weight+numel(ix);count=count+1;
 if isnan(worst),worst=abs(v);else,worst=max(worst,abs(v));end
end
if weight>0,slope=total/weight;end
end

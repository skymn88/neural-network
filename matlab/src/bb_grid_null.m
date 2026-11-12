function [root,found,count]=bb_grid_null(grid,value,preferred)
% Linear interpolation of all zero crossings; choose nearest TRAIN hold median.
grid=grid(:);value=value(:);candidates=grid(value==0);
for k=1:numel(grid)-1
 if isfinite(value(k))&&isfinite(value(k+1))&&sign(value(k))*sign(value(k+1))<0
  candidates(end+1,1)=grid(k)-value(k)*(grid(k+1)-grid(k))/(value(k+1)-value(k)); %#ok<AGROW>
 end
end
candidates=unique(candidates);count=numel(candidates);found=count>0;root=NaN;
if found,[~,k]=min(abs(candidates-preferred));root=candidates(k);end
end

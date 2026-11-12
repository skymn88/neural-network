function [center,info]=bb_train_null(tr,cfg)
% TRAIN-only fixed static pilot, independent of candidate/validation selection.
if cfg.nullGridPoints<2||cfg.nullGridPoints~=floor(cfg.nullGridPoints)||cfg.nullHoldFraction<=0||cfg.nullHoldFraction>1,error('Invalid null grid/hold fraction');end
u=[];speed=[];rangeU=[];
for f=1:numel(tr)
 rangeU=[rangeU;tr(f).S.DDV]; %#ok<AGROW>
 y=tr(f).S.RAM;rate=gradient(y)*cfg.Fs;
 width=max(1,round(cfg.dmlpRateSmoothSeconds*cfg.Fs));
 rate=conv(rate,ones(width,1)/width,'same');ix=max(width+1,cfg.evalStartIndex):numel(y)-width;
 u=[u;tr(f).S.DDV(ix)];speed=[speed;rate(ix)]; %#ok<AGROW>
end
if isempty(u),error('TRAIN too short for static-null estimation');end
[~,order]=sort(abs(speed));hold=order(1:max(1,ceil(cfg.nullHoldFraction*numel(order))));preferred=median(u(hold));
mu=mean(u);scale=max(std(u),1e-12);z=(u-mu)/scale;A=[ones(size(z)) z tanh(z) abs(z)];
R=eye(4);R(1,1)=0;w=(A'*A/numel(u)+1e-3*R)\(A'*speed/numel(u));
grid=linspace(min(rangeU),max(rangeU),cfg.nullGridPoints)';z=(grid-mu)/scale;rate=[ones(size(z)) z tanh(z) abs(z)]*w;
[center,found,count]=bb_grid_null(grid,rate,preferred);
if max(rangeU)-min(rangeU)<1e-12,found=false;end
if ~found,center=preferred;end
info=struct('Center',center,'RootFound',found,'RootCount',count,'HoldMedian',preferred,...
 'MinDDV',min(rangeU),'MaxDDV',max(rangeU),'ReferenceRAM',NaN);
% Reference position for state-dependent speed is TRAIN-only.
x=[];for f=1:numel(tr),x=[x;tr(f).S.RAM];end %#ok<AGROW>
info.ReferenceRAM=median(x);
end

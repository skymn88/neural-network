function [summary,curve]=bb_static_null(plant,tr,cfg)
% Selected-model diagnostic only; its validation-chosen root is NEVER centering input.
[~,info]=bb_train_null(tr,cfg);grid=linspace(info.MinDDV,info.MaxDDV,cfg.nullGridPoints)';
aux=struct();names={'ActuatorForce','SupplyPressure','ReturnPressure'};
for j=1:numel(names)
 if isfield(tr(1).S,names{j})
  v=[];for f=1:numel(tr),v=[v;tr(f).S.(names{j})];end %#ok<AGROW>
  aux.(names{j})=mean(v);
 end
end
drive=zeros(size(grid));net=drive;
for k=1:numel(grid),[drive(k),net(k)]=bb_static_drive(plant.rate,grid(k),info.ReferenceRAM,aux);end
[root,found,count]=bb_grid_null(grid,drive,info.HoldMedian);
[netRoot,netFound,~]=bb_grid_null(grid,net,info.HoldMedian);
zeroCurrent=bb_static_drive(plant.ddv,0,0);center=0;
if isfield(plant.rate,'ddvCenter'),center=plant.rate.ddvCenter;end
summary=table(root,found,count,info.HoldMedian,zeroCurrent,netRoot,netFound,info.ReferenceRAM,center,...
 'VariableNames',{'DriveNullDDV','DriveRootFound','DriveRootCount','TrainSlowRAMMedianDDV','DDVAtZeroCurrent','NetRateNullDDV','NetRootFound','ReferenceRAM','AppliedTrainCenter'});
pilot=info.Center;if ~info.RootFound,pilot=NaN;end
summary.TrainPilotNullDDV=pilot;summary.TrainPilotRootFound=info.RootFound;
summary.TrainDDVSpan=info.MaxDDV-info.MinDDV;
summary.LowSpeedFraction=cfg.nullHoldFraction;summary.VelocitySmoothSeconds=cfg.dmlpRateSmoothSeconds;
curve=table(grid,drive,net,'VariableNames',{'ConstantDDV','StaticDrive','NetRAMRateAtReference'});
end

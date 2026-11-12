function model=bb_train_mean(tr,cfg)
% Sample-weighted TRAIN RAM mean, independent of inference initial condition.
total=0;n=0;
for f=1:numel(tr),total=total+sum(tr(f).S.RAM);n=n+numel(tr(f).S.RAM);end
model=bb_hold_model(cfg);model.name='TrainMean';model.constantRAM=total/n;
end

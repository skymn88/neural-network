function report=bb_horizon_report(configs,va,te,cfg)
% RESET diagnostic: measured RAM only at each window start, never training/selection.
rows={};sets={va,te};splits={'validation','test'};names=fieldnames(configs);
for j=1:numel(names)
 plant=configs.(names{j});
 for s=1:2
  for f=1:numel(sets{s})
   fl=sets{s}(f);ddv=bb_predict_batch(plant.ddv,fl);
   for horizon=[2 5 10 30 Inf]
    if isinf(horizon)
     p=bb_rollout_rate(plant.rate,fl,fl.S.RAM(1),struct('DDV',ddv));mask=false(size(p));mask(cfg.evalStartIndex:end)=true;label='fullFreeRun';
    else,[p,mask]=bb_reset_rollout(plant.rate,fl,ddv,horizon,cfg.evalStartIndex);label='RESET measured RAM at window starts';end
    predictions={p,polyval(flipud(plant.angle.coefficients),p/plant.angle.scale)};
    truths={fl.S.RAM,fl.S.Deflection};stages={'RAM','Angle'};
    for stage=1:2
     y=truths{stage}(mask);error=predictions{stage}(mask)-y;rmse=sqrt(mean(error.^2));sst=sum((y-mean(y)).^2);fit=NaN;
     if sst>1e-14,fit=100*(1-sqrt(sum(error.^2)/sst));end
     rows(end+1,:)={names{j},stages{stage},splits{s},fl.name,horizon,label,numel(y),rmse,fit}; %#ok<AGROW>
    end
   end
  end
 end
end
report=cell2table(rows,'VariableNames',{'Model','Stage','Split','Flight','HorizonSeconds','Evaluation','Samples','RMSE','FitPct'});
end

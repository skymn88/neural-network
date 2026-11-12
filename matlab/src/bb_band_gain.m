function report=bb_band_gain(plant,va,te,cfg)
sets={va,te};splits={'validation','test'};report=table();
for s=1:2
 for f=1:numel(sets{s})
  fl=sets{s}(f);ddv=bb_predict_batch(plant.ddv,fl);
  component=bb_rollout_rate(plant.rate,fl,fl.S.RAM(1));
  cascade=bb_rollout_rate(plant.rate,fl,fl.S.RAM(1),struct('DDV',ddv));
  truth={fl.S.DDV,fl.S.RAM,fl.S.RAM};pred={ddv,component,cascade};stages={'DDV','RAM_Component','RAM_Cascade'};
  ix=cfg.evalStartIndex:numel(fl.t);
  for j=1:3
   t=bb_band_metrics(truth{j}(ix),pred{j}(ix),cfg.Fs);
   keys=table(repmat({fl.name},height(t),1),repmat(splits(s),height(t),1),repmat(stages(j),height(t),1),...
    'VariableNames',{'Flight','Split','Stage'});
   report=[report;keys t]; %#ok<AGROW>
  end
 end
end
end

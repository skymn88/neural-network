function wide=bb_wide_summary(summary,selection,cfg)
% All model/stage pairs; only presentation is rounded, never selection data.
keys=unique(summary(:,{'Model','Stage'}),'stable');splits={'train','validation','test'};rows=cell(0,8);
for j=1:height(keys)
 fit=nan(1,3);rmse=fit;
 for s=1:3
  z=summary(strcmp(summary.Model,keys.Model{j})&strcmp(summary.Stage,keys.Stage{j})&strcmp(summary.Split,splits{s}),:);
  if height(z)>1,error('Duplicate wide summary key');end
  allowed=cfg.printTestForAll||s<3||ismember(keys.Model{j},{selection.RAM,'HoldInitial','TrainMean'});
  if isfield(selection,'Closed'),allowed=allowed||strcmp(keys.Model{j},selection.Closed);end
  if height(z)==1&&allowed,fit(s)=z.FitPct;rmse(s)=z.RMSE;end
 end
 rows(end+1,:)=[{keys.Model{j},keys.Stage{j}},num2cell(round(fit,2)),num2cell(round(rmse,4,'significant'))]; %#ok<AGROW>
end
wide=cell2table(rows,'VariableNames',{'Model','Stage','Fit_Train','Fit_Val','Fit_Test','RMSE_Train','RMSE_Val','RMSE_Test'});
end

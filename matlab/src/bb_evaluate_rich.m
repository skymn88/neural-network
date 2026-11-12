function [details,summary]=bb_evaluate_rich(configs,tr,va,te,cfg)
sets={tr,va,te};splits={'train','validation','test'};rows={};
channels={'DDV','RAM','Deflection'};scale=zeros(1,3);start=2*cfg.inputLags+2;
if isfield(cfg,'evalStartIndex'),start=cfg.evalStartIndex;end
for j=1:3,v=[];for f=1:numel(tr),v=[v;tr(f).S.(channels{j})];end;scale(j)=std(v);end %#ok<AGROW>
labels=fieldnames(configs);
for s=1:3
    if isfield(cfg,'evalTestOnly')&&cfg.evalTestOnly&&s~=3,continue;end
    fs=sets{s};
    for f=1:numel(fs)
        fl=fs(f);idx=start:numel(fl.t);if isempty(idx),error('Holdout too short');end
        for j=1:numel(labels)
            label=labels{j};c=configs.(label);
            u=bb_predict_batch(c.ddv,fl);
            [rd,g,ramInfo]=bb_rollout_rate(c.rate,fl,fl.S.RAM(1));rc=bb_rollout_rate(c.rate,fl,fl.S.RAM(1),struct('DDV',u));
            if isfield(c.rate,'dynamicMLP'),g=ramInfo.externalInputs;end
            one=[fl.S.RAM(1);bb_ram_step(c.rate,fl.S.RAM(1:end-1),g(2:end,:))];
            if isfield(c.rate,'constantRAM'),one(:)=c.rate.constantRAM;end
            if isfield(c.rate,'ramBounds'),one=min(max(one,c.rate.ramBounds(1)),c.rate.ramBounds(2));end
            angle=polyval(flipud(c.angle.coefficients),rc/c.angle.scale);
            direct=polyval(flipud(c.angle.coefficients),fl.S.RAM/c.angle.scale);
            preds={u,rd,rc,angle,direct};steps={u,one,[],[],direct};
            stages={'DDV','RAM_Component','RAM_Cascade','Angle_Cascade','Angle_Component'};ci=[1 2 2 3 3];
            for q=1:5
                p=preds{q};o=steps{q};if ~isempty(o),o=o(idx);end
                truth=fl.S.(channels{ci(q)});
                rows=localAdd(rows,label,stages{q},splits{s},fl.name,truth(idx),p(idx),scale(ci(q)),o);
            end
        end
    end
end
details=cell2table(rows,'VariableNames',{'Model','Stage','Split','Flight','Samples','SSE','SST','FitPct','RMSE','TrainStdNRMSEpct','CoveragePct','ObservedStd','TrainStd','OneStepSSE','MaxAbsError','FinalError','MeanBias'});
summary=table();keys=unique(details(:,{'Model','Stage','Split'}),'stable');
for j=1:height(keys)
    z=details(strcmp(details.Model,keys.Model{j})&strcmp(details.Stage,keys.Stage{j})&strcmp(details.Split,keys.Split{j}),:);
    n=sum(z.Samples);sse=sum(z.SSE);sst=sum(z.SST);fit=NaN;one=NaN;
    if sst>1e-14
        fit=100*(1-sqrt(sse/sst));
        if all(isfinite(z.OneStepSSE)),one=100*(1-sqrt(sum(z.OneStepSSE)/sst));end
    end
    rmse=sqrt(sse/n);nr=NaN;if z.TrainStd(1)>1e-12,nr=100*rmse/z.TrainStd(1);end
    row=table(keys.Model(j),keys.Stage(j),keys.Split(j),fit,one,rmse,nr,sum(z.Samples.*z.CoveragePct)/n,max(z.MaxAbsError),sum(z.Samples.*z.MeanBias)/n,...
        'VariableNames',{'Model','Stage','Split','FitPct','OneStepFitPct','RMSE','TrainStdNRMSEpct','CoveragePct','MaxAbsError','MeanBias'});
    summary=[summary;row]; %#ok<AGROW>
end
end
function rows=localAdd(rows,model,stage,split,flight,y,p,scale,one)
r=bb_compare_metrics(y,p,scale);step=NaN;if ~isempty(one),step=sum((one-y).^2);end
rows(end+1,:)=[{model,stage,split,flight},num2cell(r),{step,max(abs(p-y)),p(end)-y(end),mean(p-y)}];
end

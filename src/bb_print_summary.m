function compact=bb_print_summary(summary,selection,outDir,cfg)
% Narrow screenshot-friendly output. Never omit nonfinite failures.
if nargin<3,outDir='';end
if nargin<4,cfg=bb_config();end
splits={'train','validation','test'};stages={'RAM_Cascade','Angle_Cascade'};
if cfg.diagnostics,stages={'RAM_Cascade','Angle_Cascade','DDV','RAM_Component','Angle_Component'};end
labels=unique(summary.Model,'stable');rows={};lines={};
if isfield(selection,'smokeMode')&&selection.smokeMode
    lines{end+1}='SMOKE RUN: reduced training budget, NOT model fidelity validation';
end
if isfield(selection,'source'),lines{end+1}=sprintf('Data source: %s',selection.source);end
lines{end+1}='Fit: 100 ideal | RMSE: 0 ideal, in original signal units';
if isfield(selection,'DDV'),lines{end+1}=sprintf('DDV selected: %s',selection.DDV);end
if isfield(selection,'RAM'),lines{end+1}=sprintf('RAM selected on VALIDATION: %s',selection.RAM);end
if isfield(selection,'RAM')
    ddvFit=nan(1,3);ddvRMSE=nan(1,3);
    for si=1:3
        z=summary(strcmp(summary.Model,selection.RAM)&strcmp(summary.Stage,'DDV')&strcmp(summary.Split,splits{si}),:);
        if height(z)==1,ddvFit(si)=z.FitPct;ddvRMSE(si)=z.RMSE;end
    end
    if any(~isnan(ddvFit))
        lines{end+1}=sprintf('DDV Fit Tr/Va/Te: %.1f / %.1f / %.1f',ddvFit);
        lines{end+1}=sprintf('DDV RMSE Tr/Va/Te: %.3g / %.3g / %.3g',ddvRMSE);
    end
end
for q=1:numel(stages)
    lines{end+1}='';lines{end+1}=stages{q};
    lines{end+1}=sprintf('%-19s %7s %7s %7s | %9s %9s %9s','Model','Fit Tr','Fit Va','Fit Te','RMSE Tr','RMSE Va','RMSE Te');
    for j=1:numel(labels)
        if q>2&&(~isfield(selection,'RAM')||~strcmp(labels{j},selection.RAM)),continue;end
        fits=nan(1,3);rmse=nan(1,3);
        for s=1:3
            if s==3&&~cfg.printTestForAll&&~strcmp(labels{j},'HoldInitial')&&...
                    (~isfield(selection,'RAM')||~strcmp(labels{j},selection.RAM)),continue;end
            z=summary(strcmp(summary.Model,labels{j})&strcmp(summary.Stage,stages{q})&strcmp(summary.Split,splits{s}),:);
            if height(z)>1,error('Duplicate summary key');end
            if height(z)==1,fits(s)=z.FitPct;rmse(s)=z.RMSE;end
        end
        lines{end+1}=sprintf('%-19.19s %7.1f %7.1f %7.1f | %9.3g %9.3g %9.3g',labels{j},fits,rmse);
        rows(end+1,:)=[{labels{j},stages{q}},num2cell(fits),num2cell(rmse)]; %#ok<AGROW>
    end
end
lines{end+1}='';lines{end+1}='Selected RAM and baseline diagnostics (Tr / Va / Te):';
focus={'HoldInitial'};if isfield(selection,'RAM'),focus=[{selection.RAM} focus];end
for j=1:numel(focus)
    step=nan(1,3);normError=nan(1,3);coverage=nan(1,3);
    for s=1:3
        z=summary(strcmp(summary.Model,focus{j})&strcmp(summary.Stage,'RAM_Component')&strcmp(summary.Split,splits{s}),:);
        if height(z)==1,step(s)=z.OneStepFitPct;end
        z=summary(strcmp(summary.Model,focus{j})&strcmp(summary.Stage,'RAM_Cascade')&strcmp(summary.Split,splits{s}),:);
        if height(z)==1,normError(s)=z.TrainStdNRMSEpct;coverage(s)=z.CoveragePct;end
    end
    lines{end+1}=sprintf('%s: one-step Fit %.1f / %.1f / %.1f',focus{j},step);
    lines{end+1}=sprintf('  Cascade NRMSE %.1f / %.1f / %.1f | Coverage %.1f / %.1f / %.1f',normError,coverage);
end
lines{end+1}='One-step uses measured previous RAM; free-run does not.';
compact=cell2table(rows,'VariableNames',{'Model','Stage','Fit_Train','Fit_Val','Fit_Test','RMSE_Train','RMSE_Val','RMSE_Test'});
if isfield(selection,'loo')
    compact.LOFO_RAM_RMSE=nan(height(compact),1);
    for k=1:height(selection.loo)
        ix=strcmp(compact.Model,selection.loo.Model{k})&strcmp(compact.Stage,'RAM_Cascade');
        compact.LOFO_RAM_RMSE(ix)=selection.loo.LOFO_RAM_RMSE(k);
        lines{end+1}=sprintf('%s TRAIN LOFO RAM RMSE: %.6g',selection.loo.Model{k},selection.loo.LOFO_RAM_RMSE(k));
    end
end
if isfield(selection,'LOFO_DDV_RMSE')
    compact.LOFO_DDV_RMSE=repmat(selection.LOFO_DDV_RMSE,height(compact),1);
    lines{end+1}=sprintf('TRAIN LOFO DDV RMSE (report only): %.6g',selection.LOFO_DDV_RMSE);
end
for j=1:numel(lines),fprintf('%s\n',lines{j});end
if ~isempty(outDir)
    fid=fopen(fullfile(outDir,'summary_short.txt'),'w');if fid<0,error('Cannot create short summary');end
    cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
    for j=1:numel(lines),fprintf(fid,'%s\n',lines{j});end
end
end

function layout=bb_write_workbook(file,reports,selection,cfg,timings)
% Five-sheet report, or historical named sheets. No ActiveX/Excel requirement.
names=fieldnames(reports);layout=struct();
if cfg.legacySheets
 for j=1:numel(names),writetable(reports.(names{j}),file,'UseExcel',false,'Sheet',names{j});end
 writetable(localTime(timings),file,'UseExcel',false,'Sheet','Timings');return;
end
selected=localKV(selection);flags=localKV(cfg);time=localTime(timings);
summary=cell(0,0);summary=localBlock(summary,'Selected configuration',selected);
keep={'staticNull','nullCenter','nullDDVCenter','dmlpBalancedLoss','dmlpPositionWeight','dmlpMoveRateWeight','dmlpHoldRateWeight','dmlpDisplacementWeight','selectionMode','selectionWindowSeconds','oeWindowSeconds','holdDroopMax','pulseHold','horizonFit','dynamicMLP','ddvAllSamples','fbSteadyStart','fbUniquePoles','fbTanhSearch','nlFocus','ramTrainInput','oeMaxTrainRows','nlFilterBank','ddvFilterBank','nlWiener','oeTransitionWeight','holdBaselines','nlDiagnostics','nlRefitTrainVal','legacySheets'};
summary=localBlock(summary,'Settings',flags(ismember(flags.Setting,keep),:));
layout.summaryTimeRow=size(summary,1)+3;summary=localBlock(summary,'Stage times (seconds)',time);
summary=localBlock(summary,'Fit / RMSE by model and stage',bb_wide_summary(reports.Summary,selection,cfg));
writecell(summary,file,'UseExcel',false,'Sheet','Summary','Range','A1');
per=reports.PerFlight;
if isfield(reports,'RefitPerFlight'),per=[per;reports.RefitPerFlight];end
writetable(per,file,'UseExcel',false,'Sheet','PerFlight');
searchNames={'WindowSelection','DMLP_Selection','Hold_Selection','SS_Selection','NL_Selection','OE_Selection','FB_Selection','WH_Selection','StateLeak','StructuralSelection','BetaCurve','ValidationSelection','ClosedSelection'};
search=localSearch(reports,searchNames);writecell(search,file,'UseExcel',false,'Sheet','Search','Range','A1');
skip=[searchNames {'Summary','Compact','PerFlight','RefitPerFlight','RunInfo','Split'}];diag=cell(0,0);
for j=1:numel(names)
 if ~ismember(names{j},skip),diag=localBlock(diag,names{j},reports.(names{j}));end
end
if isempty(diag),diag={'Diagnostics disabled'};end
writecell(diag,file,'UseExcel',false,'Sheet','Diagnostics','Range','A1');
config=localBlock(cell(0,0),'RunInfo',reports.RunInfo);config=localBlock(config,'Split',reports.Split);
config=localBlock(config,'cfg',flags);layout.configTimeRow=size(config,1)+3;
config=localBlock(config,'Stage times (seconds)',time);
writecell(config,file,'UseExcel',false,'Sheet','Config','Range','A1');
bb_clean_workbook(file);
end
function tab=localKV(s)
names=fieldnames(s);rows=cell(0,2);
for j=1:numel(names)
 v=s.(names{j});
 if isnumeric(v)||islogical(v),text=mat2str(v);
 elseif ischar(v),text=v;
 elseif iscell(v)&&all(cellfun(@ischar,v(:))),text=strjoin(v(:)',', ');
 else,continue;end
 rows(end+1,:)={names{j},text}; %#ok<AGROW>
end
tab=cell2table(rows,'VariableNames',{'Setting','Value'});
end
function tab=localTime(t)
names=fieldnames(t);v=zeros(numel(names),1);
for j=1:numel(names),v(j)=t.(names{j});end
tab=table(names,v,'VariableNames',{'Stage','Seconds'});
end
function data=localBlock(data,title,tab)
if width(tab)==0,tab=table({'No rows'},'VariableNames',{'Status'});end
block=[{title} repmat({''},1,max(0,width(tab)-1));tab.Properties.VariableNames;table2cell(tab)];
cols=max(size(data,2),size(block,2));if cols==0,cols=1;end
if ~isempty(data)
 if size(data,2)<cols,data(:,end+1:cols)={''};end
 data(end+1,1:cols)={''};
end
if size(block,2)<cols,block(:,end+1:cols)={''};end
data=[data;block];
end
function data=localSearch(reports,names)
headers={'Stage','Source'};
for j=1:numel(names)
 if isfield(reports,names{j}),headers=union(headers,reports.(names{j}).Properties.VariableNames,'stable');end
end
data=headers;
for j=1:numel(names)
 if ~isfield(reports,names{j}),continue;end
 t=reports.(names{j});block=repmat({''},height(t),numel(headers));
 for k=1:width(t),col=find(strcmp(headers,t.Properties.VariableNames{k}));block(:,col)=table2cell(t(:,k));end
 stage='RAM';if strcmp(names{j},'ClosedSelection'),stage='Closed';end
 if ~ismember('Stage',t.Properties.VariableNames),block(:,1)={stage};end
 block(:,2)={names{j}};data=[data;block]; %#ok<AGROW>
end
end

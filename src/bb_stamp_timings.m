function bb_stamp_timings(file,layout,timings,cfg)
% Final small update. Timings measure work up to this stamp, not this stamp itself.
names=fieldnames(timings);seconds=zeros(numel(names),1);
for j=1:numel(names),seconds(j)=timings.(names{j});end
t=table(names,seconds,'VariableNames',{'Stage','Seconds'});
if cfg.legacySheets,writetable(t,file,'UseExcel',false,'Sheet','Timings');return;end
% layout rows point at the table header (one title row precedes it).
writetable(t,file,'UseExcel',false,'Sheet','Summary','Range',sprintf('A%d',layout.summaryTimeRow));
writetable(t,file,'UseExcel',false,'Sheet','Config','Range',sprintf('A%d',layout.configTimeRow));
end

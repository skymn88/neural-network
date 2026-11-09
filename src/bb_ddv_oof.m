function [out,report]=bb_ddv_oof(tr,cfg,label)
% Each held flight is excluded from fitting AND normalization. No val/test here.
out=tr;rows={};
if numel(tr)<3
 warning('bb:OOFFallback','OOF needs at least 3 training flights; using measured DDV.');
 report=table();return;
end
old=rng;cleanup=onCleanup(@()rng(old)); %#ok<NASGU>
[c,kind]=bb_rich_config(cfg,label);
for f=1:numel(tr)
 keep=true(1,numel(tr));keep(f)=false;
 model=bb_fit_candidate(tr(keep),c,label,{'CoilCurrent'},{'DDV'},kind);
 p=bb_predict_batch(model,tr(f));out(f).S.DDV=p;
 ix=cfg.evalStartIndex:numel(p);e=p(ix)-tr(f).S.DDV(ix);
 rows(end+1,:)={tr(f).name,sqrt(mean(e.^2)),sum(e.^2),numel(e)}; %#ok<AGROW>
end
report=cell2table(rows,'VariableNames',{'Flight','LOFO_DDV_RMSE','SSE','Samples'});
end

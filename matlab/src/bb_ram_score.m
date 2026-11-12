function [score,metrics]=bb_ram_score(m,va,vDDV,cfg)
% Selection API has validation only. Test must never be passed by a search.
sse=0;n=0;wsse=0;wn=0;
extra=strcmp(cfg.selectionMode,'windowed')||any(cfg.oeWindowSeconds>0)||cfg.oeWindowSearch;
for f=1:numel(va)
 p=bb_rollout_rate(m,va(f),va(f).S.RAM(1),struct('DDV',vDDV{f}));ix=cfg.evalStartIndex:numel(p);
 sse=sse+sum((p(ix)-va(f).S.RAM(ix)).^2);n=n+numel(ix);
 if extra
  [p,mask]=bb_reset_rollout(m,va(f),vDDV{f},cfg.selectionWindowSeconds,cfg.evalStartIndex);
  wsse=wsse+sum((p(mask)-va(f).S.RAM(mask)).^2);wn=wn+sum(mask);
 end
end
full=sqrt(sse/n);window=NaN;if extra,window=sqrt(wsse/wn);end
score=full;if strcmp(cfg.selectionMode,'windowed'),score=window;end
droop=NaN;eligible=isfinite(score);
if isfinite(cfg.holdDroopMax)
 if ~isfield(cfg,'selectionPulse'),error('Pulse selection stimulus was not prepared');end
 fl=cfg.selectionPulse;p=bb_rollout_rate(m,fl,fl.S.RAM(1),struct('DDV',fl.S.DDV));
 [~,droop]=bb_pulse_metrics(fl,p,cfg.Fs);eligible=eligible&&droop<=cfg.holdDroopMax;
end
if ~eligible,score=Inf;end
metrics=struct('FullFreeRunRMSE',full,'ResetWindowRMSE',window,'HoldDroop',droop,'Eligible',eligible);
end

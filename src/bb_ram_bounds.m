function model=bb_ram_bounds(model,tr,cfg)
% TRAIN-only optional output constraint; immutable inference metadata.
if ~cfg.clampRAM,return;end
if cfg.clampMargin<0,error('clampMargin must be nonnegative');end
lo=Inf;hi=-Inf;
for f=1:numel(tr),lo=min(lo,min(tr(f).S.RAM));hi=max(hi,max(tr(f).S.RAM));end
margin=cfg.clampMargin*(hi-lo);model.ramBounds=[lo-margin hi+margin];
end

function [refit,info]=bb_refit_trainval(original,tr,va,cfg,ddvLabel)
% Refit learned weights on TRAIN+VALIDATION with frozen selected hyperparameters.
% No test data or model/degree selection is accepted by this API.
all=[tr va];timer=tic;[c,kind]=bb_rich_config(cfg,ddvLabel);
refit=original;refit.ddv=bb_fit_candidate(all,c,ddvLabel,{'CoilCurrent'},{'DDV'},kind);
rt=all;
if strcmp(cfg.ramTrainInput,'oof')
 % Time-split fragments from one source flight stay in the same OOF group.
 groups={all.name};
 if numel(tr)<=3
  groups=regexprep(groups,'_(train|validation)$','');
 end
 oc=cfg;oc.oofGroupNames=groups;[rt,~]=bb_ddv_oof(all,oc,ddvLabel);
end
m=original.rate;oc=cfg;oc.oeTransitionWeight=0;
if isfield(m,'transitionWeight'),oc.oeTransitionWeight=m.transitionWeight;end
oc.oeNonlinear=strcmp(m.inputBasis,'nonlinear');
oc.oeZeroRateAtNull=isfield(m,'noIntercept')&&m.noIntercept;
if isfield(m,'tanhScale'),oc.oeTanhScale=m.tanhScale;end
if isfield(m,'featureSpec'),oc.oeCustom=true;oc.featureSpec=m.featureSpec;end
refit.rate=bb_oe_train(rt,oc,m.beta,m.inputLags,m.ridge);refit.rate.name=m.name;
if isfield(m,'stateLeak'),refit.rate.stateLeak=m.stateLeak;end
if isfield(m,'wh'),[refit.rate,~]=bb_wh_train(refit.rate,rt,cfg,true);end
% Fixed original TRAIN clamp bounds; refit must not loosen constraints on test.
if isfield(m,'ramBounds'),refit.rate.ramBounds=m.ramBounds;end
x=[];y=[];
for f=1:numel(all),x=[x;all(f).S.RAM];y=[y;all(f).S.Deflection];end %#ok<AGROW>
scale=std(x);if scale<1e-12,error('Refit needs RAM excitation');end
A=ones(numel(x),original.angle.degree+1);
for j=1:original.angle.degree,A(:,j+1)=(x/scale).^j;end
refit.angle=struct('coefficients',A\y,'scale',scale,'degree',original.angle.degree);
info=table(numel(tr),numel(va),m.beta,m.inputLags,m.ridge,toc(timer),...
 {'Hyperparameters fixed; DDV/RAM/angle weights refit; leak and clamp fixed; no test selection.'},...
 'VariableNames',{'TrainSegments','ValidationSegments','Beta','Lags','Ridge','TrainSeconds','Note'});
end

function [model,report]=bb_wh_train(base,tr,cfg,fixedBeta)
% Frozen input model; fit monotone h knot values + beta on TRAIN free runs only.
% No validation/test arguments. No toolbox quantile/optimization dependency.
if nargin<4,fixedBeta=false;end
if isfield(base,'stateLeak'),error('NL_WH requires an NL_OE base without state leakage');end
if cfg.whKnots<3||cfg.whKnots~=floor(cfg.whKnots),error('Invalid knot count');end
model=base;rates=cell(size(tr));all=[];truth=[];
for f=1:numel(tr)
 rates{f}=bb_predict_batch(base,tr(f))*base.rateGain+base.rateBias;
 ix=cfg.evalStartIndex:numel(tr(f).t);all=[all;rates{f}(ix)];truth=[truth;tr(f).S.RAM(ix)]; %#ok<AGROW>
end
ordered=sort(all);pos=1+(numel(ordered)-1)*linspace(0,1,cfg.whKnots);
knots=unique([interp1((1:numel(ordered))',ordered,pos(:),'linear');0]);
spread=std(all);targetScale=std(truth);
if numel(knots)<3||spread<1e-12,error('NL_WH needs varying TRAIN FIR output');end
if targetScale<1e-12,targetScale=1;end
[~,anchor]=min(abs(knots));
bound=min(.95*cfg.Fs,max(cfg.whBetaLimit,abs(base.beta)*1.5));
if bound<=0,error('Invalid WH beta bound');end
p0=sqrt(diff(knots)/spread);q=min(1-1e-10,max(1e-10,-base.beta/bound));
if ~fixedBeta,p0=[p0;log(q/(1-q))];end
decode=@(p)localDecode(p,base,knots,spread,anchor,bound,fixedBeta);
objective=@(p)localLoss(decode(p),tr,rates,cfg,targetScale);
initial=objective(p0);timer=tic;
options=optimset('Display','off','MaxFunEvals',cfg.whMaxEvals,'MaxIter',cfg.whMaxIter,'TolX',1e-5,'TolFun',1e-7);
[p,loss,exitflag,output]=fminsearch(objective,p0,options);
if ~isfinite(loss)||loss>initial,p=p0;loss=initial;end
model=decode(p);model.name='NL_WH';
report=table(numel(knots),base.beta,model.beta,initial*targetScale,loss*targetScale,toc(timer),output.funcCount,exitflag,...
 'VariableNames',{'Knots','InitialBeta','Beta','InitialTrainRMSE','TrainRMSE','TrainSeconds','FunctionEvaluations','ExitFlag'});
end
function m=localDecode(p,base,knots,spread,anchor,bound,fixedBeta)
m=base;n=numel(knots);v=spread*[0;cumsum(p(1:n-1).^2)];v=v-v(anchor);
m.wh=struct('knots',knots,'values',v,'zeroAnchored',true,'extrapolation','constant');
if ~fixedBeta,m.beta=-bound/(1+exp(-max(-35,min(35,p(end)))));end
end
function score=localLoss(m,tr,rates,cfg,scale)
if any(~isfinite(m.wh.values)),score=realmax;return;end
sse=0;n=0;
for f=1:numel(tr)
 r=bb_wh_map(m.wh,rates{f});p=bb_wh_rollout(m,r,tr(f).S.RAM(1));
 ix=cfg.evalStartIndex:numel(p);e=(p(ix)-tr(f).S.RAM(ix))/scale;
 sse=sse+sum(e.^2);n=n+numel(e);
end
score=sqrt(sse/n);if ~isfinite(score),score=realmax;end
end

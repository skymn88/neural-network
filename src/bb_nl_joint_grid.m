function grid=bb_nl_joint_grid(base,cfg,seen)
% Bounded deterministic joint neighborhood; no RNG or test/validation access.
if cfg.nlJointBudget<0||cfg.nlJointBudget~=floor(cfg.nlJointBudget),error('Invalid joint budget');end
grid=zeros(0,4);multipliers=[.75 1 1.25];if cfg.nlWideBeta,multipliers=[multipliers 2 3];end
b=unique(base.beta*multipliers);if base.beta==0,b=[0 -.001 -.003];end
% Never extend common warmup/evaluation window or use more than NL's 128 lags.
limit=min([128 max(cfg.oeLags) cfg.evalStartIndex-cfg.inputLags-2]);
l=unique(max(0,min(limit,round(base.inputLags*[.5 1 1.5]))));
r=base.ridge*[.1 1 10];s=unique([base.tanhScale cfg.nlTanhScales]);
all=zeros(0,4);
for a=b,for d=l,for e=r,for f=s
 if a<=0&&1+a/cfg.Fs>0&&e>0&&f>0,all(end+1,:)=[a d e f];end %#ok<AGROW>
end;end;end;end
if isempty(all)||cfg.nlJointBudget==0,return;end
all=unique(all,'rows','stable');all=all(~ismember(all,seen,'rows'),:);
if isempty(all),return;end
% Spread the small budget over the joint Cartesian neighborhood, not one axis.
ix=unique(round(linspace(1,size(all,1),min(cfg.nlJointBudget,size(all,1)))));
grid=all(ix,:);
end

function [drive,net]=bb_static_drive(m,c,x,aux)
% Exact steady external-input features (unity-DC FB sections), no transient run.
if nargin<4,aux=struct();end
fl.t=(0:max(m.inputLags,8))'/m.Fs;fl.S=struct();names=m.inputNames;
for j=1:numel(names)
 value=0;if j==1,value=c;elseif isfield(aux,names{j}),value=aux.(names{j});end
 fl.S.(names{j})=repmat(value,numel(fl.t),1);
end
extra=fieldnames(aux);for j=1:numel(extra),fl.S.(extra{j})=repmat(aux.(extra{j}),numel(fl.t),1);end
if isfield(m,'filterBank')
 u=zeros(1,numel(names));
 for j=1:numel(names)
  u(j)=fl.S.(names{j})(1);
  if strcmp(names{j},'DDV')&&isfield(m,'ddvCenter'),u(j)=u(j)-m.ddvCenter;end
 end
 sections=m.filterBank.rawLags+1+numel(m.filterBank.poles)+size(m.filterBank.a,1);
 v=bb_forward(m,repmat(u,1,sections));
else,v=bb_predict_batch(m,fl);v=v(end,:);end
if isfield(m,'rateGain'),v=m.rateGain*v;end
if isfield(m,'rateBias'),v=v+m.rateBias;end
if isfield(m,'wh'),v=bb_wh_map(m.wh,v);end
if isfield(m,'dynamicMLP')
 net=(bb_ram_step(m,x,v)-x)*m.Fs;drive=net;
else
 drive=v;net=v;if isfield(m,'beta'),net=(bb_ram_step(m,x,v)-x)*m.Fs;end
end
end

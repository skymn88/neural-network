function idx=bb_training_indices(U,first,last,count,cfg,flightIndex)
% Input-only TRAIN sampling. Default reproduces uniform v6 selection.
valid=(first:last)';if isempty(valid),error('No valid training windows');end
count=min(count,numel(valid));mixed=isfield(cfg,'trainingSampling')&&strcmp(cfg.trainingSampling,'mixed');
if ~mixed
    idx=valid(unique(round(linspace(1,numel(valid),count))));return;
end
uniform=valid(unique(round(linspace(1,numel(valid),max(1,floor(count/2))))));
remaining=setdiff(valid,uniform);nExtra=count-numel(uniform);
if nExtra==0,idx=sort(uniform);return;end
% Base MATLAB weighted sampling without replacement (no Statistics Toolbox).
du=[0;abs(diff(U(:,1)))];score=conv(du,ones(cfg.inputLags+1,1),'same');
w=score(remaining);w=w+max(max(w)*1e-3,1e-12);
state=rng;rng(4200+flightIndex);
keys=-log(max(rand(numel(remaining),1),realmin))./w;
rng(state);[~,order]=sort(keys);
idx=sort([uniform;remaining(order(1:nExtra))]);
end

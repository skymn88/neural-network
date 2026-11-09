function r=bb_compare_metrics(y,p,trainStd)
n=numel(y); valid=isfinite(y)&isfinite(p);coverage=100*sum(valid)/n;
if all(valid),sse=sum((p-y).^2);else,sse=Inf;end
sst=sum((y-mean(y)).^2);rmse=sqrt(sse/n);
if sst<=1e-14*max(1,sum(y.^2)),fit=NaN;else,fit=100*(1-sqrt(sse/sst));end
if trainStd>1e-12,nrmse=100*rmse/trainStd;else,nrmse=NaN;end
r=[n sse sst fit rmse nrmse coverage std(y) trainStd];
end

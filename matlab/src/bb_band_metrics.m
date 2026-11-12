function out=bb_band_metrics(y,p,Fs)
% Offline FFT projection; not an inference filter. Positive lag = prediction delayed.
y=y(:);p=p(:);N=numel(y);bands=[0 .3;.3 1;1 3;3 8;8 min(32,Fs/2)];rows=[];
if N~=numel(p)||N<20,error('Band diagnostics need matching series of >=20 samples');end
freq=min((0:N-1)',N-(0:N-1)')*Fs/N;
Y=fft(y-mean(y));P=fft(p-mean(p));floorStd=max(eps,1e-10*std(y));
for b=1:size(bands,1)
 mask=freq>=bands(b,1)&freq<bands(b,2);
 if b==size(bands,1),mask=freq>=bands(b,1)&freq<=bands(b,2);end
 a=real(ifft(Y.*mask));v=real(ifft(P.*mask));sy=std(a);sp=std(v);
 gain=NaN;corr0=NaN;lagBest=NaN;gainBest=NaN;corrBest=NaN;best=-Inf;
 if sy>floorStd&&all(isfinite(v))
  gain=sp/sy;corr0=localCorr(a,v);
  % Prefer smallest |lag| when correlations tie.
  order=[0 reshape([1:8;-(1:8)],1,[])];
  for lag=order
   ix=max(1,1-lag):min(N,N-lag);yy=a(ix);pp=v(ix+lag);cc=localCorr(yy,pp);
   if isfinite(cc)&&abs(cc)>best+1e-12
    best=abs(cc);lagBest=lag;corrBest=cc;gainBest=std(pp)/std(yy);
   end
  end
 end
 rows(end+1,:)=[bands(b,:) sy sp gain corr0 lagBest gainBest corrBest]; %#ok<AGROW>
end
out=array2table(rows,'VariableNames',{'LowHz','HighHz','MeasuredStd','PredictedStd','StdRatio','Correlation','BestLagSamples','AlignedStdRatio','AlignedCorrelation'});
end
function c=localCorr(a,b)
a=a-mean(a);b=b-mean(b);den=norm(a)*norm(b);c=NaN;if den>0,c=(a'*b)/den;end
end

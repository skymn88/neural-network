function test_band_gain()
Fs=64;t=(0:4095)'/Fs;y=sin(2*pi*t);p=.4*circshift(y,3);
r=bb_band_metrics(y,p,Fs);row=find(r.LowHz==1);
assert(abs(r.StdRatio(row)-.4)<1e-12&&r.BestLagSamples(row)==3);
assert(r.AlignedCorrelation(row)>1-1e-12&&abs(r.AlignedStdRatio(row)-.4)<1e-12);
z=bb_band_metrics(ones(size(y)),zeros(size(y)),Fs);assert(all(isnan(z.StdRatio)));
c=bb_config();c.inputLags=1;c.evalStartIndex=12;f.name='bands';f.t=t;
f.S.CoilCurrent=y;f.S.DDV=y;f.S.RAM=cumsum(y)/Fs;f.S.Deflection=f.S.RAM;
plant.ddv=bb_fit_candidate(f,c,'NL_OE',{'CoilCurrent'},{'DDV'},'NL');plant.rate=bb_oe_train(f,c,-.3,1,1e-3);
r=bb_band_gain(plant,f,f,c);assert(height(r)==30&&all(ismember(r.Split,{'validation','test'})));
end

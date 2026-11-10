function test_nl_focus()
c=bb_config();c.nlJointSearch=false;c.nlRefine=true;c.evalStartIndex=10;c.oeMaxTrainRows=300;c.oeChunkRows=17;c.nlTanhScales=[.5 1];
f.name='nl';f.t=(0:149)'/64;u=sin((1:150)'*.17);f.S.DDV=u;
f.S.CoilCurrent=u;f.S.RAM=cumsum(tanh(u/.5))/64;f.S.Deflection=2*f.S.RAM;
[m,r,score]=bb_nl_search(f,f,{u},c,2,-.3,1e-3);
assert(height(r)>1&&isfinite(score));assert(score<=r.ValidationRMSE(1));
assert(abs(score-min(r.ValidationRMSE))<1e-12);
X=bb_lag_inputs(f,{'DDV'},2);z=(X-m.xMean)./m.xStd;
expected=[ones(150,1) z tanh(z/m.tanhScale) abs(z)]*m.weights*m.yStd;
assert(max(abs(bb_forward(m,X)-expected))<1e-12);
c.nlFocus=false;[~,r,~]=bb_nl_search(f,f,{u},c,2,-.3,1e-3);assert(height(r)==1);
[dc,kind]=bb_rich_config(c,'NL_OE');assert(strcmp(kind,'NL'));
ddv=bb_fit_candidate(f,dc,'NL_OE',{'CoilCurrent'},{'DDV'},kind);
pred=bb_predict_batch(ddv,f);g=f;g.S=rmfield(g.S,'DDV');
assert(isequal(pred,bb_predict_batch(ddv,g))&&all(isfinite(pred)));
assert(strcmp(ddv.inputBasis,'nonlinear'));
plant.ddv=bb_fit_candidate(f,c,'FIR',{'CoilCurrent'},{'DDV'},'FIR');plant.rate=m;
plant.angle=struct('coefficients',[0;2],'scale',1);
old=get(0,'DefaultFigureVisible');cleanup=onCleanup(@()set(0,'DefaultFigureVisible',old)); %#ok<NASGU>
set(0,'DefaultFigureVisible','off');figs=bb_plot_flight_outputs(f,plant,struct('RAM','NL_OE'),'');
assert(numel(figs)==1);assert(numel(findall(figs,'Type','axes'))==3);close(figs);
end

function test_refit_leak()
c=bb_config();c.inputLags=2;c.evalStartIndex=10;c.oeMaxTrainRows=100;c.nlRefine=false;c.leakScaleFactors=1;c.leakMaxEvals=6;
for j=1:3
 f(j).name=sprintf('flight%d',j);f(j).t=(0:99)'/64; %#ok<AGROW>
 u=sin((1:100)'*(.1+.01*j));f(j).S.CoilCurrent=u;f(j).S.DDV=u;
 f(j).S.RAM=j+cumsum(u)/64;f(j).S.Deflection=2*f(j).S.RAM;
end
[m,~,~]=bb_nl_search(f(1),f(2),{f(2).S.DDV},c,2,-1,1e-3);
[l,r,score]=bb_state_leak_fit(m,f(1),f(2),{f(2).S.DDV},c);
assert(score<=r.ValidationRMSE(1));assert(height(r)==2);
g=f(2);g.S=rmfield(g.S,'RAM');assert(isequal(bb_rollout_rate(l,g,2),bb_rollout_rate(l,f(2),2)));
a.ddv=bb_fit_candidate(f(1),c,'NL_OE',{'CoilCurrent'},{'DDV'},'NL');a.rate=m;
a.angle=struct('degree',1,'scale',1,'coefficients',[0;2]);
[b,info]=bb_refit_trainval(a,f(1),f(2),c,'NL_OE');
assert(info.TrainSegments==1&&info.ValidationSegments==1);
assert(b.rate.beta==a.rate.beta&&b.rate.inputLags==a.rate.inputLags&&b.rate.ridge==a.rate.ridge);
assert(b.angle.degree==a.angle.degree);
models.Original_TRAIN=a;models.Refit_TRAIN_VAL=b;c.evalTestOnly=true;
[d,s]=bb_evaluate_rich(models,f(1),f(2),f(3),c);
assert(all(strcmp(d.Split,'test'))&&all(strcmp(s.Split,'test')));
% All fragments of a held source flight must be excluded from OOF training.
tr=[f f];c.oofGroupNames={'A','B','C','A','B','C'};
[o,~]=bb_ddv_oof(tr,c,'NL_OE');tr(1).S.DDV=tr(1).S.DDV+10;tr(4).S.DDV=tr(4).S.DDV-10;
[q,~]=bb_ddv_oof(tr,c,'NL_OE');assert(isequal(o(1).S.DDV,q(1).S.DDV)&&isequal(o(4).S.DDV,q(4).S.DDV));
end

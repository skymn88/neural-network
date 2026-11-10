function test_nl_extensions()
c=bb_config();c.nlDiagnostics=false;c.nlJointSearch=false;c.nlRefitTrainVal=false;c.nlStateLeak=false;
c.inputLags=2;c.evalStartIndex=12;c.oeMaxTrainRows=200;c.oeLags=[2 4];c.nlRefine=false;
f.name='diagnostic';f.t=(0:119)'/64;u=sin((1:120)'*.13);
f.S.CoilCurrent=u;f.S.DDV=u;f.S.RAM=2+cumsum(u)/64;f.S.Deflection=3*f.S.RAM;
meanModel=bb_train_mean(f,c);p=bb_rollout_rate(meanModel,f,-100);
assert(all(p==mean(f.S.RAM)));assert(bb_ram_step(meanModel,999,999)==mean(f.S.RAM));
[m,r,s]=bb_nl_search(f,f,{u},c,[2 4],[-.3 -1],1e-3);
assert(~ismember('TrainSeconds',r.Properties.VariableNames));
c.nlDiagnostics=true;c.nlJointSearch=true;c.nlJointBudget=5;c.nlWideBeta=true;
[n,rr,ss]=bb_nl_search(f,f,{u},c,[2 4],[-.3 -1],1e-3);
assert(ss<=s&&sum(strcmp(rr.Search,'joint'))<=5);
assert(ismember('TrainSeconds',rr.Properties.VariableNames));
grid=bb_nl_joint_grid(m,c,cell2mat(table2cell(r(:,{'Beta','Lags','Ridge','TanhScale'}))));
assert(size(grid,1)<=5&&all(grid(:,2)<=4)&&all(grid(:,1)<=0));
plant.ddv=bb_fit_candidate(f,c,'NL_OE',{'CoilCurrent'},{'DDV'},'NL');plant.rate=n;
plant.angle=struct('coefficients',[0;3],'scale',1,'degree',1);
held=f;held.S.RAM(:)=mean(f.S.RAM);
[curve,holds,eq]=bb_nl_diagnostics(rr,plant,f,held,held,c);
assert(height(curve)==numel(unique(rr.Beta))&&height(eq)==1&&height(holds)>0);
assert(abs(eq.LinearMemorySeconds+1/(64*log(1+n.beta/64)))<1e-12);
if n.beta<0,assert(abs(n.beta*eq.EquilibriumRAM+eq.ZeroDDVRate)<1e-10);end
% Default update is exactly the legacy linear recurrence; zero-k is identical.
x=[1;2];rate=[.1;.2];expected=(1+m.beta/c.Fs)*x+rate/c.Fs;
assert(isequal(bb_ram_step(m,x,rate),expected));
m.stateLeak=struct('k',0,'center',1,'scale',2);assert(isequal(bb_ram_step(m,x,rate),expected));
m.stateLeak.k=.4;assert(max(abs(bb_ram_step(m,x,rate)-(expected-.4*tanh((x-1)/2)/64)))<1e-12);
end

function test_structural_models()
c=bb_config();c.evalStartIndex=10;c.oeMaxTrainRows=Inf;c.oeChunkRows=13;c.clampRAM=true;
f.name='structural';f.t=(0:199)'/64;u=sin((1:200)'*.13);f.S.DDV=u;
f.S.RAM=cumsum(u)/64;f.S.ActuatorForce=2+u;f.S.SupplyPressure=4+u;f.S.ReturnPressure=ones(200,1);
m.inputLags=3;m.noIntercept=false;m.featureSpec=struct('basis','deadband','parameter',.2,'force',false,'pressure',false);
m.inputNames={'DDV'};X=bb_oe_features(m,f,(1:200)');
assert(size(X,2)==4);assert(max(abs(X(:,1)-sign(u).*max(abs(u)-.2,0)))<1e-12);
assert(isequal(X(2:end,2),X(1:end-1,1)));
m.featureSpec.basis='tanh';m.featureSpec.parameter=.3;X=bb_oe_features(m,f,1);assert(abs(X(1)-tanh(u(1)/.3))<1e-12);
m.featureSpec.force=true;m.featureSpec.pressure=true;X=bb_oe_features(m,f,1);assert(size(X,2)==4+27);
g=f;g.S=rmfield(g.S,'ActuatorForce');failed=false;try,bb_oe_features(m,g,1);catch,failed=true;end;assert(failed);
c.hamDeadbands=.2;c.hamScales=[];
[m,r,score]=bb_oe_search(f,f,{u},c,'HAM_OE_F_P',3,0,1e-3);assert(height(r)==1&&isfinite(score));
bounded=bb_ram_bounds(struct(),f,c);
expected=[min(f.S.RAM) max(f.S.RAM)]+[-1 1]*c.clampMargin*(max(f.S.RAM)-min(f.S.RAM));
assert(max(abs(bounded.ramBounds-expected))<1e-12);
off=c;off.clampRAM=false;assert(isempty(fieldnames(bb_ram_bounds(struct(),f,off))));
p=bb_rollout_rate(m,f,100);assert(all(p>=m.ramBounds(1)&p<=m.ramBounds(2)));
g=f;g.S=rmfield(g.S,'RAM');assert(isequal(p,bb_rollout_rate(m,g,100)));
assert(numel(m.weights)<3*(3+1+27)+1);
raw=bb_make_demo();raw=rmfield(raw,'ActuatorForce');c.useForce=true;failed=false;
try,bb_prepare(raw(1),c);catch,failed=true;end;assert(failed);
end

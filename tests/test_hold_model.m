function test_hold_model()
c=bb_config();c.inputLags=2;c.evalStartIndex=8;c.trainStartIndex=8;
c.holdLags=[0 2];c.holdDeadbandFractions=[0 .02];c.holdRidge=1e-6;c.oeChunkRows=17;
u=zeros(240,1);u(20:40)=.2;u(120:150)=-.15;
f.name='pulse_hold';f.t=(0:239)'/64;f.S.DDV=u;
f.S.RAM=3+cumsum(2*u+.4*abs(u))/64;
[m,r,score]=bb_hold_search(f,f,{u},c);
assert(isfinite(score)&&height(r)==4&&m.beta==0&&m.noIntercept);
x=bb_rollout_rate(m,f,3);
assert(max(abs(diff(x(55:110))))<1e-12,'Zero input must retain position after finite memory');
g=f;g.S=rmfield(g.S,'RAM');assert(isequal(x,bb_rollout_rate(m,g,3)));
y=bb_rollout_rate(m,g,10);assert(max(abs(y-x-7))<1e-10,'Initial position must persist');
assert(x(60)>x(1)&&x(end)<x(110),'Pulse directions should be learned');
assert(score<.01,'Learnable pulse/hold fixture regression');
% Independent plotting entry must remain a script, with editable flight index.
script=fileread(which('plot_saved_flights'));
assert(isempty(regexp(script,'(?m)^function\s','once'))&&contains(script,'for i=1'));
end

function test_ram_loo()
c=bb_config();c.evalStartIndex=40;c.oeMaxTrainRows=100;c.inputLags=2;
for j=1:3
 f(j).name=sprintf('L%d',j);f(j).t=(0:119)'/64; %#ok<AGROW>
 u=sin((1:120)'*(.1+.03*j));f(j).S.CoilCurrent=u;f(j).S.DDV=u;f(j).S.RAM=cumsum(u)/64;
end
rates.FIR_OE=bb_oe_train(f,c,0,2,1e-3);
r=bb_ram_loo(f,c,rates,'FIR_rich');assert(height(r)==1&&isfinite(r.LOFO_RAM_RMSE));
assert(r.Samples==3*(120-40+1));
end

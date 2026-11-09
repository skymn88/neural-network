function model=bb_angle_select(tr,va,start)
% Choose degree using VALIDATION only; standardize using training RAM only.
if nargin<3,start=66;end
x=[];y=[];
for f=1:numel(tr),x=[x;tr(f).S.RAM];y=[y;tr(f).S.Deflection];end %#ok<AGROW>
scale=std(x);if scale<1e-12,error('Insufficient RAM excitation for angle fit');end
best=Inf;
for degree=[1 3]
    A=ones(numel(x),degree+1);
    for j=1:degree,A(:,j+1)=(x/scale).^j;end
    c=A\y;sse=0;n=0;
    for f=1:numel(va)
        idx=start:numel(va(f).t);if isempty(idx),error('Validation too short');end
        p=polyval(flipud(c),va(f).S.RAM(idx)/scale);
        sse=sse+sum((p-va(f).S.Deflection(idx)).^2);n=n+numel(idx);
    end
    if sse/n<best,best=sse/n;model.coefficients=c;model.scale=scale;model.degree=degree;end
end
model.validationRMSE=sqrt(best);
end

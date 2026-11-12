function Y=bb_forward(model,X)
% Toolbox-free portable forward inference for trained MLP/NARX.
if isfield(model,'dynamicMLP')
 q=model.dynamicMLP;U=bb_dmlp_inputs(X);
 Y=bsxfun(@rdivide,bsxfun(@minus,U,q.uMean),q.uStd);return;
end
if size(X,2)~=numel(model.xMean)
    error('Wrong input feature width: got %d expected %d',size(X,2),numel(model.xMean));
end
if isfield(model,'kind')
    Z=bsxfun(@rdivide,bsxfun(@minus,double(X),model.xMean),model.xStd);
    if strcmp(model.kind,'FIR')
        if isfield(model,'inputBasis')&&strcmp(model.inputBasis,'nonlinear')
            scale=1;if isfield(model,'tanhScale'),scale=model.tanhScale;end
            Z=[Z tanh(Z/scale) abs(Z)];
        end
        if ~isfield(model,'noIntercept')||~model.noIntercept,Z=[ones(size(Z,1),1) Z];end
        P=Z*model.weights;
    elseif strcmp(model.kind,'GP')
        P=zeros(size(Z,1),model.nOutputs); C=model.centers;
        for first=1:2048:size(Z,1)
            ii=first:min(first+2047,size(Z,1)); V=Z(ii,:);
            d=max(0,bsxfun(@plus,sum(V.^2,2),sum(C.^2,2)')-2*(V*C'));
            P(ii,:)=exp(-.5*d/model.lengthScale^2)*model.alpha;
        end
    else
        error('Unknown model kind');
    end
    Y=bsxfun(@plus,bsxfun(@times,P,model.yStd),model.yMean);
    return;
end
A=bsxfun(@rdivide,bsxfun(@minus,double(X),model.xMean),model.xStd).';
for k=1:numel(model.layers)
    L=model.layers(k);
    Z=bsxfun(@plus,L.W*A,L.b);
    switch L.activation
        case 'tansig'
            A=tanh(Z);
        case 'purelin'
            A=Z;
        case 'logsig'
            A=1./(1+exp(-Z));
        otherwise
            error('Unsupported activation %s',L.activation);
    end
end
Y=bsxfun(@plus,bsxfun(@times,A.',model.yStd),model.yMean);
end

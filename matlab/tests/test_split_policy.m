function test_split_policy()
% Quick split tests for Ver4.1 (no toolbox training and no result folders).
cfg=bb_config();
cfg.inputLags=4;
cfg.outputLags=3;
cfg.guardSeconds=0.2;
R=bb_make_demo();
base=bb_prepare(R(1),cfg);
for n=1:20
    flights=repmat(base,1,n);
    for i=1:n
        flights(i).name=sprintf('F%02d',i);
    end
    [tr,va,te,manifest,description]=bb_split(flights,cfg); %#ok<ASGLU>
    if n<=3
        assert(numel(tr)==n && numel(va)==n && numel(te)==n);
        for i=1:n
            assert(numel(tr(i).t)>0 && numel(va(i).t)>0 && numel(te(i).t)>0);
            assert(numel(tr(i).t)>numel(va(i).t) && numel(va(i).t)>numel(te(i).t));
        end
    else
        if n<10
            expected=[n-2 1 1];
        elseif n==11
            expected=[7 2 2];
        else
            nVal=max(1,round(.2*n)); nTest=max(1,round(.1*n));
            expected=[n-nVal-nTest nVal nTest];
        end
        assert(isequal([numel(tr) numel(va) numel(te)],expected), ...
               'Bad split at N=%d',n);
        allNames=[{tr.name} {va.name} {te.name}];
        assert(numel(unique(allNames))==n,'Flight appears in multiple splits');
    end
end
fprintf('SPLIT POLICY PASS: N=1..20. 10->7/2/1; 11->7/2/2; 20->14/4/2.\n');
end

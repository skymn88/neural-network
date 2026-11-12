function test_raw_fixture()
paths=bb_paths();d=load(paths.synthetic);
assert(numel(d.Raw)==11);ids=[d.trainIDs(:);d.validationIDs(:);d.testIDs(:)];
assert(numel(unique(ids))==11&&isequal(sort(ids),(1:11)'));
assert(numel(d.trainIDs)==7&&numel(d.validationIDs)==2&&numel(d.testIDs)==2);
durations=[500 550 600 650 700 750 800 900 1000 1100 850];
names={'CoilCurrent','DDV','RAM','Deflection'};
for f=1:11
 for j=1:numel(names)
    s=d.Raw(f).(names{j});assert(numel(s.x)==durations(f)*64+1);
    assert(all(isfinite(s.x))&&all(isfinite(s.t))&&all(diff(s.t)>0));
    assert(max(abs(diff(s.t)-1/64))<1e-9);
 end
end
fprintf('RAW FIXTURE: 11 FLIGHTS / 64 HZ / SPLIT CHECKS PASS\n');
end

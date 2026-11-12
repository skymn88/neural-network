function out=bb_plot_user_vs_synthetic(syntheticScale,syntheticOffset)
if nargin<1,syntheticScale=[1 1 1 1];end
if nargin<2,syntheticOffset=[0 0 0 0];end
assert(numel(syntheticOffset)==4&&all(isfinite(syntheticOffset)));
assert(numel(syntheticScale)==4&&all(isfinite(syntheticScale))&&all(syntheticScale~=0));
% One figure per paired flight, four channels, common 64-Hz sample index.
paths=bb_paths();folder=paths.results;d=load(paths.synthetic);
if ~exist(folder,'dir'),mkdir(folder);end
U=load_user_flights();cfg=bb_config();cfg.requireForce=false;cfg.usePressure=false;
n=min(numel(U),numel(d.Raw));
out=fullfile(folder,['input_comparison_' datestr(now,'yyyymmdd_HHMMSS')]);
if exist(out,'dir'),out=tempname(folder);end;mkdir(out);
channels={'CoilCurrent','DDV','RAM','Deflection'};
for i=1:n
    real=bb_prepare(U(i),cfg);syn=bb_prepare(d.Raw(i),cfg);
    h=figure('Name',sprintf('Flight %d: user vs synthetic',i),'Color','w','Position',[60 60 1200 850]);
    N=min(numel(real.t),numel(syn.t));
    for j=1:4
        subplot(4,1,j);name=channels{j};a=real.S.(name);b=syn.S.(name)*syntheticScale(j)+syntheticOffset(j);
        plot(0:N-1,a(1:N),'k','LineWidth',1);hold on;
        plot(0:N-1,b(1:N),'r-','LineWidth',.8);grid on;ylabel(name,'Interpreter','none');
        if j==1,title(sprintf('Flight %d | %s vs %s | first %d samples',i,real.name,syn.name,N),'Interpreter','none');legend('User','Synthetic');end
        if j==4,xlabel('Sample index at 64 Hz (no time warping / amplitude scaling)');end
    end
    savefig(h,fullfile(out,sprintf('flight_%02d.fig',i)));
    print(h,fullfile(out,sprintf('flight_%02d.png',i)),'-dpng','-r140');
end
fprintf('%d paired flight figures: %s\n',n,out);
fprintf('Units must match: synthetic current A, DDV normalized stroke, RAM m, angle deg.\n');
if numel(U)~=numel(d.Raw),fprintf('Unpaired flights: user %d, synthetic %d. Only first %d paired.\n',numel(U)-n,numel(d.Raw)-n,n);end
end

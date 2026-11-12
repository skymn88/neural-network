function [A,state]=bb_oe_window_filter(Z,idx,a,Fs,W,state)
% Reset ONLY the integrated design at global window starts, not input features.
if W<2||W~=floor(W)||numel(idx)~=size(Z,1)||any(diff(idx)~=1),error('Invalid window filter input');end
A=zeros(size(Z));if isempty(state),state=zeros(1,size(Z,2));end
first=1;
while first<=numel(idx)
 start=floor((idx(first)-1)/W)*W+1;
 last=min(numel(idx),first+(start+W-1-idx(first)));
 block=Z(first:last,:);
 if idx(first)==start,state(:)=0;block(1,:)=0;end
 [A(first:last,:),state]=filter(1/Fs,[1 -a],block,state,1);
 first=last+1;
end
end

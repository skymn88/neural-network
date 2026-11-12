function y=bb_wh_map(map,r)
% Monotone piecewise linear static map, constant saturation outside knot range.
shape=size(r);r=r(:);z=min(max(r,map.knots(1)),map.knots(end));
y=interp1(map.knots,map.values,z,'linear');y=reshape(y,shape);
end

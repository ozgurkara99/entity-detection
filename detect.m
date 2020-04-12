function [flag_north, flag_south, counted_n, counted_s] = detect(pos, rad, coord)
%DETECT Summary of this function goes here
%   Detailed explanation goes here
dist = (pos(:,1)-coord(1)).^2 + (pos(:,2)-coord(2)).^2 ;
flag_north = find(dist<rad.^2 & pos(:,2)>coord(2));
flag_south = find(dist<rad.^2 & pos(:,2)<coord(2));
counted_n = length(find(dist<rad.^2 & pos(:,2)>coord(2)));
counted_s = length(find(dist<rad.^2 & pos(:,2)<coord(2)));
end


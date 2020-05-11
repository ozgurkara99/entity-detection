function [flag_north, flag_south, counted_n, counted_s] = detect(pos, rad, coord)

dist = (pos(:,1)-coord(1)).^2 + (pos(:,2)-coord(2)).^2 ; %find the euclidian distance
flag_north = find(dist<rad.^2 & pos(:,2)>coord(2)); %detect the indices of the molecules that passed the north 
flag_south = find(dist<rad.^2 & pos(:,2)<coord(2)); %detect the indices of the molecules that passed the south 
counted_n = length(find(dist<rad.^2 & pos(:,2)>coord(2))); %detect the number of the molecules that passed the north 
counted_s = length(find(dist<rad.^2 & pos(:,2)<coord(2))); %detect the number of the molecules that passed the south 
end


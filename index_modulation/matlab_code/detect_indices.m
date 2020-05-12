function [indices] = detect_indices(pos,radius,coord)
%{
    Detect the indices of values in pos which are inside of the sphere that is defined
    by given radius and center.

    Parameters
    ----------
    pos : array
        the position matrix of molecules
    radius : float
        the radius of sphere
    coord : array
        the center point of sphere [x,y,z]
    Returns
    -------
    array
        the indices of the "pos" array that are detected inside the sphere
%}

    distance = euclidian_dist(pos, coord);
    indices = find(distance < radius);

end


function [arr] = mirror_point_over_plane(a, b, c, d, x1, y1, z1)
%{
    Mirror given point (x1,y1,z1) over given plane equation ax + by + cz + d = 0

    Parameters
    ----------
    a, b, c, d : float
        plane equation coefficients
    x1, y1, z1 : float
        the point coordinates that will be mirrored
    Returns
    -------
    array
        the coordinates of reflected point [x,y,z]
%}   

    k =(-a * x1 - b * y1 - c * z1 - d)/(a * a + b * b + c * c); 
    x2 = a * k + x1 ;
    y2 = b * k + y1 ;
    z2 = c * k + z1 ;
    x3 = 2 * x2-x1 ;
    y3 = 2 * y2-y1 ;
    z3 = 2 * z2-z1 ;
    arr = [x3 y3 z3];

end


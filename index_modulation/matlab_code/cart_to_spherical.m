function [az,el] = cart_to_spherical(x,y,z)
%{
    Convert the given cartesian coordinate system to spherical coordinate system
    ex: (0,0,1) -> elevation is 0
    ex: (1,0,0) -> azimuth is 0

    Parameters
    ----------
    x : float
        x coordinate of point
    y : float
        y coordinate of point
    z : float
        z coordinate of point
    Returns
    -------
    az : float
        azimuth value
    el : float
        elevation value
%}
    
    hxy = sqrt(x.^2 + y.^2);
    el = atan2(hxy, z) ;
    az = atan2(y, x);
    
end


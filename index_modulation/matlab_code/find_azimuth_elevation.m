function [azimuth, elevation] = find_azimuth_elevation(coords,center_of_sphere)
%{
    """Find the azimuth and elevation values of "coords" array with 
    respect to sphere centered at center_of_sphere 

    Parameters
    ----------
    coords : array
        the coordinates array of molecules that will be converted to spherical coordinates
    center_of_sphere : array
        the center point of sphere
    Returns
    -------
    azimuth : float
        the azimuth value
    elevation : float
        the elevation value
%}   

    [azimuth, elevation] = cart_to_spherical(coords(:,1) - center_of_sphere(1), coords(:,2) - center_of_sphere(2), coords(:,3) - center_of_sphere(3));
end


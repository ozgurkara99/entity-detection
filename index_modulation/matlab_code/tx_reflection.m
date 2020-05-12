function [reflected] = tx_reflection(t0, t1, center_of_tx, r)
%{
    Find the reflection point of the line segment which is crossing through t0 and t1 and 
    intersects with the sphere that is defined by parameters. Then returns this 
    reflected point
    Parameters
    ----------
    t0, t1 : array
        the points through which the line segment is crossing 
    center_of_tx : array
        the center point of sphere [x,y,z]
    r : float
        the radius value of array
    Returns
    -------
    array
        the coordinates of reflected point [x,y,z]
%}         

    coords = find_with_quadratic(t0, t1, center_of_tx, r);
    %ax + by + cz + d = 0
    a = coords(1) - center_of_tx(1);
    b = coords(2) - center_of_tx(2);
    c = coords(3) - center_of_tx(3);
    d = -1 * (a * coords(1) + b * coords(2) + c * coords(3));
    %reflect t1 point over the plane ax + by + cz + d = 0
    reflected = mirror_point_over_plane(a, b, c, d, t1(1), t1(2), t1(3));
end


function [root] = find_with_quadratic(t0,t1,center_of_sphere,r)
%{
    Find the intersection point of the line segment that is crossing through 
    t0 and t1, and the sphere which is defined by center_of_sphere and r (radius) value
    x = x0 + (x1 - x0) * t
    y = y0 + (y1 - y0) * t
    z = z0 + (z1 - z0) * t

    (x0 + (x1 - x0) * t)^2 + (y0 + (y1 - y0) * t)^2 + (z0 + (z1 - z0) * t)^2 - r^2 = 0

    Parameters
    ----------
    t0, t1 : array
        the points through which the line segment is crossing 
    center_of_sphere : array
        the center point of sphere [x,y,z]
    r : float
        the radius value of array
    Returns
    -------
    array
        the point of intersection [x,y,z]
%}          

    t0 = t0 - center_of_sphere;
    t1 = t1 - center_of_sphere;

    % f = lambda x: (coefs[0]) + (coefs[1] * x) + (coefs[2] * x**2) - coefs[3]
    e = sum(t0.^2, 2); %t0[0] ** 2 + t0[1] ** 2 + t0[2] ** 2 
    b = 2 * sum((t0 .* (t1-t0)), 2); %2 * (t0[0] * (t1[0] - t0[0]) + t0[1] * (t1[1] - t0[1]) + t0[2] * (t1[2] - t0[2]))
    a = sum((t1-t0).^2, 2);%(t1[0] - t0[0]) ** 2 + (t1[1] - t0[1]) ** 2 + (t1[2] - t0[2]) ** 2 
    d = r.^2;
    c = e - d;
    
    delta = b.^2 - 4 * a .* c;
    xn = (-1*b + sqrt(delta) )./(2*a);
    xn2 = (-1*b - sqrt(delta) )./(2*a);
    %fprintf("xn = %.4f , xn2 = %.4f",xn,xn2);
    root1 = center_of_sphere + t0 +  xn .* (t1 - t0);
    root2 = center_of_sphere + t0 +  xn2 .* (t1 - t0);
    if(sum((root1-t0).^2,2) >= sum((root2-t0).^2,2))
        root = root2;
    else
        root = root1;
    end
%{   
    root1 = center_of_sphere + np.transpose(np.array([t0[0] + (t1[0] - t0[0]) * xn, t0[1] + (t1[1] - t0[1]) * xn , t0[2] + (t1[2] - t0[2]) * xn]))
    root2 = center_of_sphere + np.transpose(np.array([t0[0] + (t1[0] - t0[0]) * xn2, t0[1] + (t1[1] - t0[1]) * xn2 , t0[2] + (t1[2] - t0[2]) * xn2]))
    if(np.sum((root1-t0)**2,axis=-1) >= np.sum((root2-t0)**2,axis=-1)):
        return root2
    else:
        return root1
%}
end


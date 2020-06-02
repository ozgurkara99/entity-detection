import numpy as np

PI = np.pi

def parametrize(t0, t1, p):
    
    """Create a new point on a line segment that is created by given t0 and t1 parameters with 
    increment p

    Parameters
    ----------
    t0 : array
        first point [x,y,z]
    t1 : array
        second point [x,y,z]
    p : float
        increment number
    Returns
    -------
    array
        new point on this line segment (x,y,z)
    """
    
    par = np.transpose(t0) + np.transpose(t1 - t0) * p
    return par

def euclidian_dist(arr1, arr2):
    
    """Find the euclidian distance between 2 points

    Parameters
    ----------
    arr1 : array
        first point [x,y,z]
    arr2 : array
        second point [x,y,z]
    Returns
    -------
    float
        the euclidian distance between points
    """
    
    return np.sqrt(abs(np.sum(((arr1 - arr2) ** 2), axis=-1)))

def cart_to_spherical(x, y, z):
    
    """Convert the given cartesian coordinate system to spherical coordinate system
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
    """
    
    hxy = np.hypot(x, y)
    el = np.arctan2(hxy, z) 
    az = np.arctan2(y, x)
    return az, el

def mirror_point_over_plane(a, b, c, d, x1, y1, z1):  
    
    """Mirror given point (x1,y1,z1) over given plane equation ax + by + cz + d = 0

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
    """    

    k =(-a * x1 - b * y1 - c * z1 - d)/float((a * a + b * b + c * c)) 
    x2 = a * k + x1 
    y2 = b * k + y1 
    z2 = c * k + z1 
    x3 = 2 * x2-x1 
    y3 = 2 * y2-y1 
    z3 = 2 * z2-z1 
    return np.stack((x3,y3,z3), axis=-1)

def delete_duplicated(pos, pos2):
    
    """Find the set difference of pos2 from pos

    Parameters
    ----------
    pos : array
        first array
    pos2 : array
        second array
    Returns
    -------
    array
        the values of pos2/pos
    """       
    
    differ = np.array(list(set(pos2).difference(set(pos))))
    return differ
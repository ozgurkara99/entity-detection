function [dist] = euclidian_dist(arr1,arr2)
%{
    Find the euclidian distance between 2 points

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
    
%}    
dist = sqrt(sum((arr1 - arr2).^2 , 2));

end


function [differ] = delete_duplicated(pos,pos2)
%{
    
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
%}     
    
    %differ = np.array(list(set(pos2).difference(set(pos))));
    differ = setdiff(pos2,pos);
end


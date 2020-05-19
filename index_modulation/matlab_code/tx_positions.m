function [arr] = tx_positions(center_of_UCA,num_of_tx, d_yz, r_tx)
%{
    """Create the transmitter's coordinates in a circular region and distributed 
    uniformly in angle depending on the self.num_of_tx

    Returns
    -------
    array
        the coordinates of created transmitters
%}          

    d = transpose(linspace(0,1-(1/num_of_tx),(num_of_tx)));
    theta = d * 2 * pi;
    y = center_of_UCA(2) + (d_yz + r_tx) * cos(theta);
    z = center_of_UCA(3) + (d_yz + r_tx) * sin(theta);
    x = ones(num_of_tx,1) * center_of_UCA(1);
    arr = [x y z];
end


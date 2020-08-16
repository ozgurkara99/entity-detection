num_of_tx=8;
num_of_rx=1;
r_rx=5;
r_tx=0.5;
D=79.4;
step=0.001;
time=5;
d_yz=10;
d_x=10;
center_of_rx = [0 0 0];
center_of_UCA = [center_of_rx(1) + d_x + r_rx +  r_tx, center_of_rx(2), center_of_rx(3)];
mu = 0;
sigma = sqrt(2 * D * step);
txpos = tx_positions(center_of_UCA,num_of_tx, d_yz, r_tx);
n_train = 1250;
fprintf("Simulation is starting...\n")

tri = zeros(8,8,n_train);

aran = [1,2,3,4,5,6,7,8];

output = [];
output_coordinates = [];
flag = 1;
mols = 500:500:4000;
error_rate = zeros(size(mols,1), 2);
moll = 0;
myarr = transpose(aran) * ones(1,n_train);
for mol_number=mols
    moll = moll + 1;
    filename = "output_" + string(mol_number) + ".csv";
    for j=1:num_of_tx
        for x = 1:n_train
            pos = repmat(txpos(j,:),mol_number,1);
            for i = 1: (time / step)

                delta = (randn(mol_number,3) + mu) * sigma;
                pos2 = pos + delta;

                %reflect over tx block
                tx_block_indices = find(pos2(:,1) >= (center_of_UCA(1) + r_tx));
                pos2(tx_block_indices,:) = mirror_point_over_plane(1, 0, 0, -1 * (center_of_UCA(1) + r_tx), pos2(tx_block_indices,1), pos2(tx_block_indices,2), pos2(tx_block_indices,3));


                %reflect over tx spheres
                for each_tx = 1:num_of_tx
                    center_of_tx = txpos(each_tx,:);
                    tx_indices_1 = detect_indices(pos, r_tx, center_of_tx);
                    tx_indices_2 = detect_indices(pos2, r_tx, center_of_tx);
                    tx_indices = delete_duplicated(tx_indices_1, tx_indices_2);
                    if(size(tx_indices,1)~=0 && size(tx_indices,2)~=0)
                        for h = 1:size(tx_indices,1)
                            c = tx_indices(h);
                            pos2(c,:) = tx_reflection(pos(c,:), pos2(c,:), center_of_tx, r_tx);  
                        end
                    end
                end
                %scatter3(pos(:,1), pos(:,2), pos(:,3));

                %detect rx
                rx_indices = detect_indices(pos2, r_rx, center_of_rx);
                if(size(rx_indices,1) ~= 0 && size(rx_indices,2)~=0)    
                    coords = [];
                    for h = 1:size(rx_indices,1)
                        y = rx_indices(h);
                        coords = [coords ; find_with_quadratic(pos(y,:), pos2(y,:), center_of_rx, r_rx)];
                    end
                    [azimuth_data, elevation_data] = find_azimuth_elevation(coords, center_of_rx);
                    for az_i = 1:size(azimuth_data,1)
                        aci = atan2(coords(az_i,2), coords(az_i,3));
                        if(aci < pi/8 && aci >= -pi/8)
                            tri(j,3,x) = tri(j,3,x) + 1;
                        elseif (aci >= pi/8 && aci < 3*pi/8)
                            tri(j,2,x) = tri(j,2,x) + 1;
                        elseif (aci >= 3*pi/8 && aci < 5*pi/8)
                            tri(j,1,x) = tri(j,1,x) + 1;
                        elseif (aci >= 5*pi/8 && aci < 7*pi/8)
                            tri(j,8,x) = tri(j,8,x) + 1;
                        elseif (aci >= 7*pi/8 && aci <= pi) || ( aci < -7*pi/8)
                            tri(j,7,x) = tri(j,7,x) + 1;
                        elseif (aci >= -7*pi/8 && aci < -5*pi/8)
                            tri(j,6,x) = tri(j,6,x) + 1;  
                        elseif (aci >= -5*pi/8 && aci < -3*pi/8)
                            tri(j,5,x) = tri(j,5,x) + 1;
                        elseif (aci >= -3*pi/8 && aci < -pi/8)
                            tri(j,4,x) = tri(j,4,x) + 1;                                                                    
                        end
                    end
                    pos2(rx_indices,1) = -1000;
                    output_coordinates = [output_coordinates;coords];
                    timex = ones(size(azimuth_data,1), size(azimuth_data,2)) .* (i*step);
                    output = [output; [azimuth_data, elevation_data,timex]];
                end
                pos = pos2;
            end
            wrt = transpose(output);
            wrt = [wrt [j;j;j]];
            output = [];
            output_coordinates = [];
            if flag == 1 
                dlmwrite(filename, wrt); %create the csv file and write to it
                flag=2;
            else
                dlmwrite(filename, wrt, '-append'); %append to the created csv file
            end

        end
    end
    [M,I]=max(tri,[],1);
    xy = reshape(I,[size(I,2), size(I,3)]);
    error = sum(xy == myarr, 'all');
    error = (n_train * num_of_tx) - error;
    error_rate(moll,1) = mol_number;
    error_rate(moll,2) = error/(8*n_train);
    tri = zeros(8,8,n_train);
    if(moll == 1)
        dlmwrite('error_rate2.txt',error_rate);
    else
        dlmwrite('error_rate2.txt',error_rate(moll,:), '-append');
    end
end



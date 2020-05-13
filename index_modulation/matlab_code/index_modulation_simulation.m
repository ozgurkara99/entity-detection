num_of_tx=8;
num_of_rx=1;
r_rx=5;
r_tx=0.5;
D=79.4;
step=0.0001;
time=0.75;
d_yz=10;
d_x=10;
center_of_rx = [0 0 0];
mol_number=10000;
center_of_UCA = [center_of_rx(1) + d_x + r_rx +  r_tx, center_of_rx(2), center_of_rx(3)];
mu = 0;
sigma = sqrt(2 * D * step);
txpos = tx_positions(center_of_UCA, num_of_tx);
n_train = 150;
fprintf("Simulation is starting...")


output = [];
output_coordinates = [];
flag = 1;
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
                pos2(rx_indices,1) = -1000;
                output_coordinates = [output_coordinates;coords];
                time = ones(size(azimuth_data,1), size(azimuth_data,2)) .* (i*step);
                output = [output; [azimuth_data, elevation_data,time]];
            end
            pos = pos2;
        end
        wrt = transpose(output);
        wrt = [wrt [j;j;j]];
        output = [];
        output_coordinates = [];
        if flag == 1 
            dlmwrite('output.csv', wrt); %create the csv file and write to it
            flag=2;
        else
            dlmwrite('output.csv', wrt, '-append'); %append to the created csv file
        end
        
    end
end
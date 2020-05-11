clear;

%for 10 pairs and 1 training, it works in 18-19 seconds

mol_num = 10000; %mol number 
r_rx = 4; %radius of the receiver
coord_rx = [10,0]; %[x,y] %coordinates of the receiver
r_eve = 4;%radius of the eve
coord_eve = [100,4]; %coordinates of the eve
time_ = 5; %time of simulation
interval_length = 0.001; %discrete time steps of the simulation that occurs in each time step
down_sampled_length = 0.01; %the value of the down sampled time length 
each = time_/interval_length; %time step number 
diffusion_coeff = 79.4; %diffusion coefficient and variance, mean values
sigma = sqrt(2*diffusion_coeff*interval_length); %variance
mu = 0; %mean

initial_mol_coord = [0,0]; %initial coordinates of moles

n_train = 1; %train number 

y_rx = zeros(each,4);

%given this interval, create pairs of all coordinates with the increment
interval_x = [-2.875 10.875];
interval_y = [-11.875 11.875];
increment = 0.25;
pairs = create_pairs(interval_x, interval_y, increment);

pairs((pairs(:,1)-coord_rx(1)).^2 + (pairs(:,2)-coord_rx(2)).^2  < (r_rx + r_eve).^2, :)=[]; %discarding overlapped situations
size_of_pairs = size(pairs,1); %the size of pairs
output = zeros(n_train,1003); %create the temporary output variable which is written to csv file
for k=1:size_of_pairs %for each of the pairs
    coord_eve = pairs(k,:); %assign coordinates of eve to selected pair
    for j=1:n_train %for given pair, in each training 
        pos = repmat(initial_mol_coord,mol_num,1); %create a position matrix with (mol_num x 2) dimensions
        for i = 1:each %for each time step
            delta = (randn(mol_num,2) + mu) * sigma; %in each time step, molecules move gaussian distributed in each axis depending on th diffusion coefficient etc.
            pos = pos + delta; %update the positions of molecules
            [flag_rx_n, flag_rx_s,  counted_rx_n, counted_rx_s] = detect(pos, r_rx, coord_rx); %detect the indices of the molecules which pass through receiver and count the number
            [flag_eve_n, flag_eve_s, counted_eve_n, counted_eve_s] = detect(pos, r_eve, coord_eve); %detect the indices of the molecules which pass through eve and count the number
            y_rx(i,1:4) = [i * interval_length , counted_rx_n , counted_rx_s, counted_rx_n + counted_rx_s]; %y_rx is a 2D matrix with (each,4) dimensions. 
            %It stores [time, the number of molecules passed south of rx,
            %the number of molecules passed north of rx, sum of south and
            %north]
            pos([flag_rx_n;flag_rx_s;flag_eve_n;flag_eve_s],2)=10000; %teleport the molecules which passed rx or eve, far away
        end
        out_rx = downsample(y_rx(:,:), time_, down_sampled_length, interval_length); %downsample the values 
        %out_eve = downsample(y_eve(k,j,:,:), time_, down_sampled_length, interval_length);
        output(j,:) = [transpose(out_rx(:,2)) transpose(out_rx(:,3)) 1 coord_eve]; %output (160,1003) dimensions: north, south, probability, coordinates of eve
    end
    if k == 1 
        dlmwrite('output.csv', output); %create the csv file and write to it
    else
        dlmwrite('output.csv', output, '-append'); %append to the created csv file
    end
end



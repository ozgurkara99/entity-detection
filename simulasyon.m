clear;
mol_num = 10000;

r_rx = 4;
coord_rx = [10,0]; %[x,y,z]

r_eve = 4;
coord_eve = [100,4];

time_ = 5;
interval_length = 0.001;
down_sampled_length = 0.01;
each = time_/interval_length;
diffusion_coeff = 79.4;
sigma = sqrt(2*diffusion_coeff*interval_length);
mu = 0;
initial_mol_coord = [0,0];

n_train = 120;
y_rx = zeros(each,4);
%y_eve = zeros(3,n_train, each,4);

interval_x = [-2.875 10.875];
interval_y = [-11.875 11.875];
increment = 0.25;

pairs = create_pairs(interval_x, interval_y, increment);
pairs((pairs(:,1)-coord_rx(1)).^2 + (pairs(:,2)-coord_rx(2)).^2  < (r_rx + r_eve).^2, :)=[]; %discarding overlapped situations
size_of_pairs = size(pairs,1);;
%size_of_pairs = 1;
output = zeros(n_train,1003);

for k=1:size_of_pairs
    coord_eve = pairs(k,:);
    for j=1:n_train
        pos = repmat(initial_mol_coord,mol_num,1);
        for i = 1:each
            delta = (randn(mol_num,2) + mu) * sigma;
            pos = pos + delta;
            [flag_rx_n, flag_rx_s,  counted_rx_n, counted_rx_s] = detect(pos, r_rx, coord_rx);
            [flag_eve_n, flag_eve_s, counted_eve_n, counted_eve_s] = detect(pos, r_eve, coord_eve);
            y_rx(i,1:4) = [i * interval_length , counted_rx_n , counted_rx_s, counted_rx_n + counted_rx_s]; 
            %y_eve(k,j,i,1:4) = [i * interval_length , counted_eve_n , counted_eve_s, counted_eve_n + counted_eve_s];
            pos([flag_rx_n;flag_rx_s;flag_eve_n;flag_eve_s],2)=10000;
        end
        out_rx = downsample(y_rx(:,:), time_, down_sampled_length, interval_length);
        %out_eve = downsample(y_eve(k,j,:,:), time_, down_sampled_length, interval_length);
        output(j,:) = [transpose(out_rx(:,2)) transpose(out_rx(:,3)) 1 coord_eve]; %north, south, probability, coordinates of eve
    end
    if k == 1 
        dlmwrite('output.csv', output);
    else
        dlmwrite('output.csv', output, '-append');
    end
end



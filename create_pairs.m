function [arr] = create_pairs(interval_x, interval_y, increment)

vec_x = interval_x(1):increment:interval_x(2);
vec_y = interval_y(1):increment:interval_y(2);


A = vec_x.' * ones(1,size(vec_y,2));
x_vec = reshape(A.',size(A,1)*size(A,2),1);

A = vec_y.' * ones(1,size(vec_x,2));
y_vec = reshape(A,size(A,1)*size(A,2),1);

arr = [x_vec y_vec];
end


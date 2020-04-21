# Bogazici NRG, Entity Detection Project

## For More Information
[Machine Learning-Based Silent Entity Localization Using Molecular Diffusion](https://ieeexplore.ieee.org/document/8964317)

## Parameters:
Batch size: 1024  
Epoch Num: 500  
Median of l2norm prob: 0.72  
%10 percent of the dataset is used as test set  
%9 percent of the dataset is used as validation set  

The network has 1000 input and has 6 hidden layers each have 512 neurons. L2 regularization is used for overcoming the overfitting. Learning rate is chosen to decrease to half of it's initial value (0.001) hyperbolically after 500 epoch. Elu activation function is used for neurons in hidden layers. There are 2 outputs: the coordinates and probability. "Coordinates" output has no activation function since it tries to solve a regression problem, however "probability" output has a sigmoid activation function to predict the probability. 

There are 2 loss functions defined: binary cross entropy and custom mean squared error. Binary cross entropy is used for the probability layer, custom MSE is used for the coordinates layer. If true-probability of eve being is 0, then the corresponding coordinates' loss is not considered and not summed with the total loss.

Tensorboard logs will be uploaded soon

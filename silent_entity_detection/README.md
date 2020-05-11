# Bogazici NRG, Silent Entity Detection Project
## Installation
Localization of a silent entity in 2-D given scenario

MATLAB file for simulation, Python file for localization task

You should run simulation.m file (you can change it's parameters as written inside the file) to obtain an output.csv file which contains a matrix with dimensions (simulation_number x 1003). First 1000 column is the counted number, 1001th column for probability for EVE being (1 or 0), 1002th and 1003th columns are the coordinates of the EVE in 2D (x,y).

You should run the main.py file in the directory that contains "data" and "logs" folder. In data folder, there are csv files that extracted from MATLAB simulation; and logs folder for Tensorboard which will be updated soon.

## Details of Dataset Obtained by MATLAB Simulation:
One simulation will run in between 1.8-1.9 seconds.
The parameters can be changed before training. Default parameters are given below:
mol number = 10000  
radius of the receiver = 4  
coordinates of the receiver = [10 0]  
radius of the eve = 4  
time of simulation = 5  
delta time = 0.001  
down sampled time = 0.01  
diffusion coefficient = 79.4  
initial mol coordinates = [0 0]  
number of training = 100  
the region where eve can exist  
-interval_x = [-3 11]  
-interval_y = [-12 12]  
increment = 0.25  

## Details of Artificial Neural Network Model:
Batch size: 1024  
Epoch Num: 500  
Mean of l2norm prob: 0.598
%3 percent of the dataset is used as test set  
%9 of the dataset is used as validation set  

The network has 1000 input and has 6 hidden layers each have 512 neurons. L2 regularization is used for overcoming the overfitting. Learning rate is chosen to decrease to half of it's initial value (0.001) hyperbolically after 500 epoch. Elu activation function is used for neurons in hidden layers. There are 2 outputs: the coordinates and probability. "Coordinates" output has no activation function since it tries to solve a regression problem, however "probability" output has a sigmoid activation function to predict the probability. 

There are 2 loss functions defined: binary cross entropy and custom mean squared error. Binary cross entropy is used for the probability layer, custom MSE is used for the coordinates layer. If true-probability of eve being is 0, then the corresponding coordinates' loss is not considered and not summed with the total loss.

Tensorboard logs will be uploaded soon

## Results 
Predictions on test set and the results
![Predictions on test set and the results](https://github.com/ozgurkara99/entity-detection/blob/master/silent_entity_detection/images/Results.png)
Learning curve:
will be updated

## For More Information
[Machine Learning-Based Silent Entity Localization Using Molecular Diffusion](https://ieeexplore.ieee.org/document/8964317)

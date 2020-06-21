# Bogazici NRG, Index Modulation Project
## Installation
Machine Learning based decoding input (symbol) in Multiple Input Single Output scenario.

There is one simulation code written in MATLAB and program for machine learning training and testing.
Simulation will output a csv file that has (number of training x 3) rows. For one simulation, it outputs the time series of the azimuth and elevation values of molecules that are absorbed by Rx sphere.  

You should run preprocess_and_train.py file in the directory that contains data/output.csv . This file converts the simulation output to multivariate time series, then train it with CNN model to classify which Tx sent the information.  

## Simulation Parameters
Number of Tx spheres = 8 (distributed equally on the circle that has center (center_of_UCA) and radius = d_yz +  r_tx)  
Number of Rx spheres = 1  
Radius of Rx = 5  
Radius of Tx = 0.5  
Diffusion Coefficient = 79.4  
Timestep = 0.0001  
Total Time = 1  
d_yz = Distance between center of UCA and the closest point of Tx sphere = 10  
dx = Distance between center of UCA and the closest point of Rx sphere = 10  
Center of Rx = [0 0 0]  
Molecule Number = 100000  
Center of UCA = [center_of_rx(1) + d_x + r_rx +  r_tx, center_of_rx(2), center_of_rx(3)]  
Mu = 0  
Sigma = sqrt(2 * D * step)  
Number of Simulation = 150  
Tx sphere's emit molecules and also they can reflect molecules.   
Rx sphere absorb molecules.  
This simulation outputs the azimuth and elevation values of molecules that are absorbed by Rx sphere.   

## Images From Simulation
![1](https://github.com/ozgurkara99/entity-detection/blob/master/index_modulation/images/sim.png)
![2](https://github.com/ozgurkara99/entity-detection/blob/master/index_modulation/images/sim2.png)
![3](https://github.com/ozgurkara99/entity-detection/blob/master/index_modulation/images/sim3.png)
![4](https://github.com/ozgurkara99/entity-detection/blob/master/index_modulation/images/sim4.png)  

## Details of Preprocessing
For one simulation, we have 2 time series: Azimuth value of absorbed molecule/time, elevation value of absorbed molecule/time. The program first slice the sphere into (azimuth_slice x elevation_slice) segment. Then for each region, it creates new time series. 
In the preprocessing, the timeseries also downsampled to 0.0001 -> 0.01.

## Details of Deep Learning Based Model
Data splitted to training, validation and test set. (0.4, 0.3, 0.3) Then model has 2 Convolutional layer (kernel size is 2x2 and activation function is relu) and one dense layer which has activation function as softmax (because it is classification, we need to find the probabilites). (after convolution layers, data will be flattened). Because it is a multiclass classification problem using multivariate time series, "categorical crossentropy" loss function is used. "Accuracy" metric is used for evaluation. 5 epoch is selected and it is sufficient for accuracy = 1 classification.  

## Results
![Learning Curve](https://github.com/ozgurkara99/entity-detection/blob/master/index_modulation/images/Learning%20Curve.png)
![Model Accuracy](https://github.com/ozgurkara99/entity-detection/blob/master/index_modulation/images/Model%20Accuracy.png)
![Error - Mol Number](https://github.com/ozgurkara99/entity-detection/blob/master/index_modulation/images/error%20-%20mol_number.png)

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

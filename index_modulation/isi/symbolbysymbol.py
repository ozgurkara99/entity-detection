import numpy as np
import csv
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow import keras
from os import listdir
from os.path import isfile, join
import random

PI = np.pi
pi = np.pi
time = 5
downsample = 0.0001
az_pair, el_pair = 4, 4
test_size = 0.2
filepath = 'new_data\\training_data\\output_100000.csv'
folder_path = 'new_data\\training_data\\'
test_path = "new_data\\test_data\\"
size = time / downsample
mol_num = 100000
onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
"""
1 -> theta yani azimuth
2 -> phi yani elevation
"""




def sph2cart(az, el, r):
    rsin_theta = r * np.sin(el)
    y = rsin_theta * np.sin(az)
    x = rsin_theta * np.cos(az)
    z = r * np.cos(el)
    return x, y, z


def between(value, a, b):
    pos_a = value.find(a)
    if pos_a == -1: 
        return ""
    pos_b = value.rfind(b)
    if pos_b == -1: 
        return ""
    adjusted_pos_a = pos_a + len(a)
    if adjusted_pos_a >= pos_b: 
        return ""
    return value[adjusted_pos_a:pos_b]
       
def az_el_pair(az_num, el_num):
    azimuth = np.linspace(-PI, PI, az_num + 1)
    elevation = np.linspace(0,PI, el_num + 1)
    return azimuth, elevation

def read_new(folder_path):
    tri2 = []
    files_list = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for i in files_list:
        if(i[-1] != "v"):
            continue
        tri = read_data(folder_path + i)
        tri2.extend(tri)
    return tri2

def preprocess2(az_pair, el_pair,size, time, downsample, tri):
    output = np.zeros((len(tri),8 + 1,int(size)))
    print(output.shape)
    output[:,8,:] = np.linspace(0,time - downsample,int(size))
    classes = np.zeros((len(tri),1))
    for x,lis in enumerate(tri):
        i = 0
        for timex in range(int(size)):
            while((timex * downsample) <= lis[2,i] and (lis[2,i] < (timex + 1) * downsample) and (i <= lis.shape[1]-2)):
                _, y, z = sph2cart(lis[0,i], lis[1,i], 5)
                aci = np.arctan2(y,z)
                if(aci < pi/8 and aci >= -pi/8):
                    output[x,2,timex] += 1
                elif (aci >= pi/8 and aci < 3*pi/8):
                    output[x,1,timex] += 1
                elif (aci >= 3*pi/8 and aci < 5*pi/8):
                    output[x,0,timex] += 1
                elif (aci >= 5*pi/8 and aci < 7*pi/8):
                    output[x,7,timex] += 1
                elif (aci >= 7*pi/8 and aci <= pi) or ( aci < -7*pi/8):
                    output[x,6,timex] += 1
                elif (aci >= -7*pi/8 and aci < -5*pi/8):
                    output[x,5,timex] += 1 
                elif (aci >= -5*pi/8 and aci < -3*pi/8):
                    output[x,4,timex] += 1
                elif (aci >= -3*pi/8 and aci < -pi/8):
                    output[x,3,timex] += 1 
                i += 1
        classes[x,0] = lis[0,-1] 
    return classes, output

def read_data(filepath):
    tri = []
    temp = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        i = 1
        for row in reader:
            temp.append(row)
            if(i%3==0):
                tri.append(np.array(temp,dtype=float))
                temp = []
            i=i+1
            if(i==-1): #3n + 1 for result
                break
    return tri

tri = read_new(folder_path)
classes, output = preprocess2(az_pair, el_pair, size, time, downsample, tri[850:852])
x = np.zeros((1205))
for i in range(len(tri)):
    x[i] = tri[i][0,-1]
output_1 = output[0]/mol_num
clas_1 = classes[0]/mol_num
cs = np.cumsum(output_1[:-1,:], axis=1)
for i in range(8):
    plt.plot(np.arange(0,5,0.0001), cs[i,:], label=str(i))
plt.legend()

time_window = 1.0
index = int(time_window/downsample) 
col = int(time/time_window)
h_channel = np.zeros((8,col))
molecule_number = 100000


for i in range(col):
    h_channel[:,i] = cs[:,(i+1)*index-1] - cs[:,i*index]

h_3d_channel = np.zeros((8,8,col))
permutation = np.array([0, 1, 2, 3, 4, 5, 6, 7])
for i in range(8):
    h_3d_channel[i] =  h_channel[permutation, :]
    for h in range(len(permutation)):
        permutation[h] = (permutation[h] - 1) % 8




        
ts = time_window     
data_dict = {}
data_folder = "window_" + str(ts) + "_upper_4\\"
data_dict = {}
shap = [0]
for i in range(0,5):
    data_dict["x_" + str(i)] = np.load(data_folder + "data_reshaped_" + str(i) + ".npy")
    shap.append(data_dict["x_" + str(i)].shape[0] + shap[i])
    print(data_dict["x_" + str(i)].shape[0], shap[i], shap, i)
    data_dict["y_" + str(i)] = np.load(data_folder + "classes_reshaped_" + str(i) + ".npy")  

data_reshaped = np.zeros((shap[-1], 8, 50, 1))
classes = np.zeros((shap[-1], int(time/ts)))
for i in range(0,5):
    data_reshaped[shap[i]:shap[i+1],:,:,:] = data_dict["x_" + str(i)]
    classes[shap[i]:shap[i+1], :] = data_dict["y_" + str(i)]
    
example_data = np.squeeze(data_reshaped)[0]    
example_classes = classes[0]-1
simulation = list(example_classes)


L = 4 #memory size
s_i = molecule_number  # si
mu_past = np.zeros((8,len(simulation))) # simulation classes need to start from 0
var_past = np.zeros((8,len(simulation)))
#mu_present = np.zeros((8,len(simulation)))
#var_present = np.zeros((8,len(simulation)))
for k in range(len(simulation)):
    starting = 0 if k<L else k-L+1
    print("starting point:", starting,"k:",int(simulation[k]))
#    mu_present[:,k] = s_i * h_3d_channel[int(simulation[k]),:,0]
#    var_present[:,k] += s_i * np.multiply(h_3d_channel[int(simulation[k]),:,0], 1 - h_3d_channel[int(simulation[k]),:,0]) 
    for z in range(starting,k):
        class_of = int(simulation[z])
        print("class",class_of)
        print("k-z", k-z)
        mu_past[:,k] += s_i * h_3d_channel[class_of,:,k-z]
        var_past[:,k] += s_i * np.multiply(h_3d_channel[class_of,:,k-z], 1 - h_3d_channel[class_of,:,k-z]) 

mu = np.zeros((8,8,len(simulation)))
var = np.zeros((8,8,len(simulation)))
for i in range(8):
    for j in range(8):
        for k in range(len(simulation)):
            mu[i,j,k] = mu_past[j,k] + s_i * h_3d_channel[i,j,k]
            var[i,j,k] = var_past[j,k] + s_i * (1-h_3d_channel[i,j,k])



H = np.zeros((8,8,len(simulation)))
received_mol = np.zeros((8,len(simulation)))
for k in range(len(simulation)):
    received_mol[:,k] = np.sum(example_data[:,k*int(ts/0.1):((k+1)*int(ts/0.1))-1], axis=1)*molecule_number
for i in range(8):
    for j in range(8):
        for k in range(len(simulation)):
            H[i,j,k] = np.log(1/(np.sqrt(2*np.pi*var[i,j,k]))) - np.square(received_mol[j,k] - mu[i,j,k]) / (2*var[i,j,k])

index_k = np.zeros((1,len(simulation)))
H_summed = np.sum(H,axis=0)
estimated = np.argmax(abs(H_summed), axis=0)


#for k in range(len(simulation)):
#    mu_k = mu[:,k]
#    var_k = var[:,k]
#    received_k = np.sum(example_data[:,k*int(ts/0.1):((k+1)*int(ts/0.1))-1], axis=1)*molecule_number
#    cost = np.log(1/(np.sqrt(2*np.pi*var_k))) - (np.square(received_k - mu_k) / 2*var_k)
#    




import numpy as np
import csv
import math
PI = math.pi
time = 0.75
downsample = 0.01
size = time / downsample
filepath = 'data\\output.csv'

"""
1 -> theta yani azimuth
2 -> phi yani elevation
"""
az_pair, el_pair = 4, 4
def az_el_pair(az_num, el_num):
    azimuth = np.linspace(-PI, PI, az_num + 1)
    elevation = np.linspace(0,PI, el_num + 1)
    return azimuth, elevation

def preprocess(arr):
    pass
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
            if(i==31):
                break
    return tri


tri = read_data(filepath)

output = np.zeros((len(tri),az_pair * el_pair + 1,int(size)))
output[:,az_pair * el_pair,:] = np.linspace(0,time - downsample,int(size))

az_list, el_list = az_el_pair(az_pair, el_pair)
for inn,lis in enumerate(tri):
    for time in range(int(size)):    
        az = lis[0,np.where(((time * downsample) <= lis[2,:]) & (lis[2,:]< (time +1)* downsample))]
        el = lis[1,np.where(((time * downsample) <= lis[2,:]) & (lis[2,:]< (time +1)* downsample))]
        for i in range(az.shape[1]):
            for j in range(az_pair):
                for k in range(el_pair):
                    if((az_list[j] <= az[0,i] < az_list[j+1]) & (el_list[k] <= el[0,i] < el_list[k+1])):
                        index = j * az_pair + k
                        output[inn,index,time] = output[inn,index,time] + 1

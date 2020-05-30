import numpy as np
import csv
import math
PI = math.pi

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
            if(i==10):
                break
    return tri


tri = read_data(filepath)

output = np.zeros((len(tri),17,750))
output[:,16,:] = np.linspace(0,0.749,750)

az_list, el_list = az_el_pair(az_pair, el_pair)
for inn,lis in enumerate(tri):
    for time in range(750):    
        az = lis[0,np.where(((time * 0.001) <= lis[2,:]) & (lis[2,:]< (time +1)* 0.001))]
        el = lis[1,np.where(((time * 0.001) <= lis[2,:]) & (lis[2,:]< (time +1)* 0.001))]
        for i in range(az.shape[1]):
            for j in range(az_pair):
                for k in range(el_pair):
                    if((az_list[j] <= az[0,i] < az_list[j+1]) & (el_list[k] <= el[0,i] < el_list[k+1])):
                        index = j * az_pair + k
                        output[inn,index,time] = output[inn,index,time] + 1

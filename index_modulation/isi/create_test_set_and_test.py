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
import pickle

#a = {'hello': 'world'}


PI = np.pi
pi = np.pi
time = 5
downsample = 0.1
az_pair, el_pair = 4, 4
test_size = 0.2
filepath = 'new_data\\training_data\\output_100000.csv'
folder_path = 'new_data\\training_data\\'
test_path = "new_data\\test_data\\"
size = time / downsample
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

def plot_error(data, label, logarithmic=True):
    plt.title("Error - Molecule number")
    if(logarithmic): 
        
        plt.semilogy(data[1,:]/3, data[2,:], "-o", label=label)
    else:
        plt.scatter(data[1,:]/3,data[2,:], "-o", label=label)
    plt.xlabel("Normalized Molecule Number")
    plt.ylabel("Error")
    plt.legend(loc=3)
    plt.grid()

def preprocess2(size, time, downsample, tri, classes):
    output = np.zeros((len(tri),8 + 1,int(size)))
    print("preprocess2 output shape:",output.shape)
    output[:,8,:] = np.linspace(0,time - downsample,int(size))
    for x,lis in enumerate(tri):
        if(x%100==0):
            print(str(x) + "/" + str(len(tri)))
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

def change_data_and_class_to_masked(data, classes):
    data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))
    new_data = np.zeros((data.shape[0]*numberr, data.shape[1], data.shape[2]))
    new_classes = np.zeros((classes.shape[0]*numberr, 1))
    for i in range(data.shape[0]):
        for j in range(numberr):
            sub = data[i,:,0:(window*(j+1))]
            k = numberr*i + j
            new_data[k,:,0:(window*(j+1))] = sub
            new_classes[k,0] = classes[i,j] 
    return np.reshape(new_data, (new_data.shape[0], new_data.shape[1], new_data.shape[2], 1)), new_classes

def natural_coding(classes, predicted):
    code_dict = {"1.0": np.array([0,0,0]),
                 "2.0": np.array([0,0,1]),
                 "3.0": np.array([0,1,0]),
                 "4.0": np.array([0,1,1]),
                 "5.0": np.array([1,0,0]),
                 "6.0": np.array([1,0,1]),
                 "7.0": np.array([1,1,0]),
                 "8.0": np.array([1,1,1])}
    classes = np.reshape(classes, (classes.shape[0] * classes.shape[1], 1))
    predicted = np.reshape(predicted, (predicted.shape[0] * predicted.shape[1], 1))
#    new_classes = np.zeros(classes, (classes.shape[0], 3))
#    new_predicted = np.zeros(predicted, (predicted.shape[0], 3))
    total = classes.shape[0] * 3
    true_total = 0
    for i in range(classes.shape[0]):
        true_total += np.sum(code_dict[str(classes[i,0])] == code_dict[str(predicted[i,0])])
    return 1 - true_total/total


def gray_coding(classes, predicted):
    code_dict = {"1.0": np.array([0,0,0]),
                 "2.0": np.array([0,0,1]),
                 "3.0": np.array([0,1,1]),
                 "4.0": np.array([0,1,0]),
                 "5.0": np.array([1,1,0]),
                 "6.0": np.array([1,1,1]),
                 "7.0": np.array([1,0,1]),
                 "8.0": np.array([1,0,0])}
    classes = np.reshape(classes, (classes.shape[0] * classes.shape[1], 1))
    predicted = np.reshape(predicted, (predicted.shape[0] * predicted.shape[1], 1))
#    new_classes = np.zeros(classes, (classes.shape[0], 3))
#    new_predicted = np.zeros(predicted, (predicted.shape[0], 3))
    total = classes.shape[0] * 3
    true_total = 0
    for i in range(classes.shape[0]):
        true_total += np.sum(code_dict[str(classes[i,0])] == code_dict[str(predicted[i,0])])
    return 1 - true_total/total

def select_data(number, data_reshaped, classes):
    new_data = data_reshaped[:,:,:(number+1)*window,:]
    new_classes = classes[:,number]
    return new_data, new_classes   
import os 
"""
real_res_dict = {}


for numberr in range(10,1,-1):
    window_size = time / numberr 
    window = int(window_size*10)
    upper_limit = min(4,numberr)      
    data_folder = "window_" + str(window_size) + "_upper_" + str(upper_limit) + "\\"
    res_dict = {}    
    files_list = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    res_naive = np.zeros((4,len(files_list)))
    res_model = np.zeros((4,len(files_list)))
    res_masked = np.zeros((4,len(files_list)))
    res_multi_model = np.zeros((4,len(files_list)))
    data_folder = "window_" + str(window_size) + "_upper_" + str(upper_limit) + "\\"
    model_multi = keras.models.load_model(data_folder + "multi_output_model.h5")
    model_masked = keras.models.load_model(data_folder + "data_masked.h5")
    
    #model = keras.models.load_model("1_layer_256_epoch1200.h5")
    for index,file in enumerate(files_list):
        mol_number = int(between(file, "_", "."))
        print(file)
        file2 = test_path + file
        tri = read_data(file2)
        isi_data = []
        test_size = 2000
        isi_data_y = np.zeros((test_size, numberr))
        for j in range(0,test_size):
            z = np.zeros((1,numberr))
            if(j%100==0):
                print("j equals to: " + str(j) + "/" + str(test_size))
            one_example_y = np.zeros((1,numberr))
            for i in range(0,numberr):
                rand2 = random.randint(0,len(tri)-1)
                x = tri[rand2]
                class_of_x = int(x[-1,-1])
                one_example_y[0,i] = class_of_x
                if(i == 0):
                    isi_data.append(np.squeeze(x[:,np.where((x[2,:-1] + ((i)*window_size) < (window_size*(i+upper_limit))) & (x[2,:-1] + ((i)*window_size) < (time)))]))
                else:
                    length = len(np.where((x[2,:-1] + ((i)*window_size) < (window_size*(i+upper_limit))) & (x[2,:-1] + ((i)*window_size) < (time)))[0])
                    temp = np.zeros((3,length))
                    row = np.ones((1,length)) * ((i)*window_size)
                    temp[2,:] = row
                    y = np.squeeze(x[:,np.where((x[2,:-1] + ((i)*window_size) < (window_size*(i+upper_limit))) & (x[2,:-1] + ((i)*window_size) < (time)))]) + temp
                    #print(y.shape, i)
                    #y = y.reshape(y.shape[0], y.shape[2])
                    isi_data[j] = np.hstack((isi_data[j], y))
            isi_data_y[j,:] = one_example_y
            isi_data[j] = isi_data[j][ :, isi_data[j][2].argsort()]
        classes, data = preprocess2(size, time, downsample, isi_data, isi_data_y) 
        data_reshaped = data[:,:-1,:]
        data_reshaped = data_reshaped / mol_number
        data_reshaped = data_reshaped.reshape(data_reshaped.shape[0],data_reshaped.shape[1],data_reshaped.shape[2],1)
        
        predicted_out = np.zeros((classes.shape[0], classes.shape[1]))
        for ii in range(numberr):
            new_data_reshaped, new_classes = select_data(ii, data_reshaped, classes)
            
            multi_model = keras.models.load_model(data_folder + "multi_model_" + str(ii) + ".h5")
            res1 = multi_model.predict(new_data_reshaped)
            res2 = np.argmax(res1, axis=1)
            res3 = res2 + 1
            predicted_out[:,ii] = res3
        print("multi model:", 1 - (np.sum(classes == predicted_out))/(classes.shape[0]*classes.shape[1])) 
        print("multi model natural coding:", natural_coding(classes, predicted_out))  
        print("multi model gray coding:", gray_coding(classes, predicted_out)) 
        res_multi_model[0,index] = 1 - (np.sum(classes == predicted_out))/(classes.shape[0]*classes.shape[1])
        res_multi_model[1,index] = int(mol_number)
        res_multi_model[2,index] = natural_coding(classes, predicted_out)
        res_multi_model[3,index] = gray_coding(classes, predicted_out)
        
        predicted_out_naive = np.zeros((classes.shape[0], classes.shape[1]))
        for ii in range(numberr):
            new_data = np.squeeze(data_reshaped[:,:,(ii*window):((ii+1)*window)])
            h = np.argmax(np.sum(new_data, axis=2), axis=1)+1
            predicted_out_naive[:,ii] = h
        print("naive:", 1 - (np.sum(classes == predicted_out_naive))/(classes.shape[0]*classes.shape[1])) 
        print("naive natural coding:", natural_coding(classes, predicted_out_naive))  
        print("naive gray coding:", gray_coding(classes, predicted_out_naive))         
        res_naive[0,index] = 1 - (np.sum(classes == predicted_out_naive))/(classes.shape[0]*classes.shape[1])
        res_naive[1,index] = int(mol_number)
        res_naive[2,index] = natural_coding(classes, predicted_out_naive)
        res_naive[3,index] = gray_coding(classes, predicted_out_naive)
        
        res = model_multi.predict(data_reshaped)
        predicted_out = np.zeros((classes.shape[0], classes.shape[1]))
        for aaa,el in enumerate(res):
            b = np.argmax(el, axis=1)
            b = b + 1
            predicted_out[:,aaa] = b
        print("multi output model:", 1 - (np.sum(classes == predicted_out))/(classes.shape[0]*classes.shape[1]))  
        print("multi output natural coding:", natural_coding(classes, predicted_out))  
        print("multi output gray coding:", gray_coding(classes, predicted_out))        
        res_model[0,index] = 1 - (np.sum(classes == predicted_out))/(classes.shape[0]*classes.shape[1])
        res_model[1,index] = int(mol_number)    
        res_model[2,index] = natural_coding(classes, predicted_out)
        res_model[3,index] = gray_coding(classes, predicted_out)
        
        a,b = change_data_and_class_to_masked(data_reshaped, classes)
        h = model_masked.predict(a)
        hh = np.argmax(h, axis=1)
        hh = hh + 1
        res_masked[0,index] = 1 - (np.sum(np.squeeze(b) == hh))/(b.shape[0]*b.shape[1])
        res_masked[1,index] = int(mol_number)   
        res_masked[2,index] = natural_coding(hh.reshape(hh.shape[0],1).astype(float), b)
        res_masked[3,index] = gray_coding(hh.reshape(hh.shape[0],1).astype(float), b)
        
        print("masked model:", 1 - (np.sum(hh == np.squeeze(b)))/(b.shape[0]*b.shape[1]))
        print("masked model natural coding:", natural_coding(hh.reshape(hh.shape[0],1).astype(float), b))  
        print("masked model gray coding:", gray_coding(hh.reshape(hh.shape[0],1).astype(float), b))   
        
    res_dict["naive"] = res_naive
    res_dict["multi_output_model"] = res_model
    res_dict["data_masked"] = res_masked
    res_dict["multi_model"] = res_multi_model
        
    real_res_dict[str(window_size)] = res_dict
#    _ = plot_error(res_dict["multi_model"], "multi_model")
#    _ = plot_error(res_dict["multi_output_model"], "multi_output_model")
#    _ = plot_error(res_dict["naive"], "naive")
#    plt2 = plot_error(res_dict["data_masked"], "data_masked")
#    
#    plt.title("Symbol, Window size: " + str(window_size))
#
#    if window_size==1.25:    
#        plt.semilogy(res_dict["multi_model"][1,:]/3,res_dict["multi_model"][2,:], "-o", label="multi_model")
#    
#    else:
#        plt.semilogy(res_dict["multi_output_model"][1,:]/3,res_dict["multi_output_model"][2,:], "-o", label="multi_output_model")
#    plt.semilogy(res_dict["naive"][1,:]/3,res_dict["naive"][2,:], "-o", label="naive")
#    #plt.semilogy(res_dict["data_masked"][1,:]/3,res_dict["data_masked"][0,:], "-o", label="data_masked")
#    
#    plt.xlabel("Normalized Molecule Number")
#    plt.ylabel("Bit Error Rate")
#    plt.legend(loc=3)
#    plt.grid()
#    plt.savefig("result_natural_coding_naive_model" + str(window_size) + ".jpg")
#    plt.clf()
#
#    plt.title("Gray Coding, Window size: " + str(window_size))
#
#    #plt.semilogy(res_dict["multi_model"][1,:]/3,res_dict["multi_model"][3,:], "-o", label="multi_model")
#    plt.semilogy(res_dict["multi_output_model"][1,:]/3,res_dict["multi_output_model"][3,:], "-o", label="multi_output_model")
#    plt.semilogy(res_dict["naive"][1,:]/3,res_dict["naive"][3,:], "-o", label="naive")
#    #plt.semilogy(res_dict["data_masked"][1,:]/3,res_dict["data_masked"][3,:], "-o", label="data_masked")
#    
#    plt.xlabel("Normalized Molecule Number")
#    plt.ylabel("Bit Error Rate")
#    plt.legend(loc=3)
#    plt.grid()
#    plt.savefig("result_gray_coding_naive_model" + str(window_size) + ".jpg")    
#    plt.clf()
    
with open('results.pickle', 'wb') as handle:
    pickle.dump(real_res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

with open('results.pickle', 'rb') as handle:
    b = pickle.load(handle)

sel_mol = 2250
ind = np.where(b[list(b.keys())[0]]["multi_output_model"][1] == sel_mol)[0][0]
tb_multi = np.zeros((8,4))
tb_naive = np.zeros((8,4))
del b["2.5"]
for i,key in enumerate(b):
    
    tb_multi[i,0] = float(key)
    tb_naive[i,0] = float(key)
    if(key == str(5/4)):
        tb_multi[i,1] = b[key]["multi_model"][0,ind] #symbol
        tb_multi[i,2] = b[key]["multi_model"][2,ind] #natural
        tb_multi[i,3] = b[key]["multi_model"][3,ind] #gray       
        
        tb_naive[i,1] = b[key]["naive"][0,ind] #symbol
        tb_naive[i,2] = b[key]["naive"][2,ind] #natural
        tb_naive[i,3] = b[key]["naive"][3,ind] #gray     
        
    else:
        tb_multi[i,1] = b[key]["multi_output_model"][0,ind] #symbol
        tb_multi[i,2] = b[key]["multi_output_model"][2,ind] #natural
        tb_multi[i,3] = b[key]["multi_output_model"][3,ind] #gray
        
        tb_naive[i,1] = b[key]["naive"][0,ind] #symbol
        tb_naive[i,2] = b[key]["naive"][2,ind] #natural
        tb_naive[i,3] = b[key]["naive"][3,ind] #gray        


plt.title("Symbol Error Rate")

plt.semilogy(tb_naive[:,0]/3,tb_naive[:,1], "-o", label="Naive Approach")
plt.semilogy(tb_naive[:,0]/3,tb_multi[:,1], "-o", label="Machine Learning Model")
   
plt.xlabel("Transmission time for bit")
plt.ylabel("Symbol Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("symbol_error_rate_tb.jpg")    
plt.clf()

plt.title("Natural Coding Bit Error Rate")

plt.semilogy(tb_naive[:,0]/3,tb_naive[:,2], "-o", label="Naive Approach")
plt.semilogy(tb_naive[:,0]/3,tb_multi[:,2], "-o", label="Machine Learning Model")
   
plt.xlabel("Transmission time for bit")
plt.ylabel("Bit Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("natural_error_rate_tb.jpg")    
plt.clf()


plt.title("Gray Coding Bit Error Rate")

plt.semilogy(tb_naive[:,0]/3,tb_naive[:,3], "-o", label="Naive Approach")
plt.semilogy(tb_naive[:,0]/3,tb_multi[:,3], "-o", label="Machine Learning Model")
   
plt.xlabel("Transmission time for bit")
plt.ylabel("Bit Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("gray_error_rate_tb.jpg")    
plt.clf()    

hhh  = str(5/3)    

#plot for window size 0.5
plt.title("Symbol Error Rate for Window Size 0.5")

plt.semilogy(b["0.5"]["naive"][1,:]/3,b["0.5"]["naive"][0,:], "-o", label="Naive Approach")
plt.semilogy(b["0.5"]["multi_output_model"][1,:]/3,b["0.5"]["multi_output_model"][0,:], "-o", label="Machine Learning Model")
   
plt.xlabel("Molecule Number for Bit (Total Molecule Number/3)")
plt.ylabel("Symbol Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("symbol_error_window_05.jpg")    
plt.clf()    

plt.title("Natural Coding Bit Error Rate for Window Size 0.5")

plt.semilogy(b["0.5"]["naive"][1,:]/3,b["0.5"]["naive"][2,:], "-o", label="Naive Approach")
plt.semilogy(b["0.5"]["multi_output_model"][1,:]/3,b["0.5"]["multi_output_model"][2,:], "-o", label="Machine Learning Model")
   
plt.xlabel("Molecule Number for Bit (Total Molecule Number/3)")
plt.ylabel("Bit Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("natural_coding_error_window_05.jpg")    
plt.clf()       


plt.title("Gray Coding Error Rate for Window Size 0.5")

plt.semilogy(b["0.5"]["naive"][1,:]/3,b["0.5"]["naive"][3,:], "-o", label="Naive Approach")
plt.semilogy(b["0.5"]["multi_output_model"][1,:]/3,b["0.5"]["multi_output_model"][3,:], "-o", label="Machine Learning Model")
   
plt.xlabel("Molecule Number for Bit (Total Molecule Number/3)")
plt.ylabel("Bit Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("gray_coding_error_window_05.jpg")    
plt.clf()           


#plot for window size 1.66    
plt.title("Symbol Error Rate for Window Size 1.66")

plt.semilogy(b[hhh]["naive"][1,:]/3,b[hhh]["naive"][0,:], "-o", label="Naive Approach")
plt.semilogy(b[hhh]["multi_output_model"][1,:]/3,b[hhh]["multi_output_model"][0,:], "-o", label="Machine Learning Model")
   
plt.xlabel("Molecule Number for Bit (Total Molecule Number/3)")
plt.ylabel("Symbol Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("symbol_error_window_1_66.jpg")    
plt.clf()    

plt.title("Natural Coding Bit Error Rate for Window Size 1.66")

plt.semilogy(b[hhh]["naive"][1,:]/3,b[hhh]["naive"][2,:], "-o", label="Naive Approach")
plt.semilogy(b[hhh]["multi_output_model"][1,:]/3,b[hhh]["multi_output_model"][2,:], "-o", label="Machine Learning Model")
   
plt.xlabel("Molecule Number for Bit (Total Molecule Number/3)")
plt.ylabel("Bit Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("natural_coding_error_window_1_66.jpg")    
plt.clf()       


plt.title("Gray Coding Error Rate for Window Size 1.66")

plt.semilogy(b[hhh]["naive"][1,:]/3,b[hhh]["naive"][3,:], "-o", label="Naive Approach")
plt.semilogy(b[hhh]["multi_output_model"][1,:]/3,b[hhh]["multi_output_model"][3,:], "-o", label="Machine Learning Model")
   
plt.xlabel("Molecule Number for Bit (Total Molecule Number/3)")
plt.ylabel("Bit Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("gray_coding_error_window_1_66.jpg")    
plt.clf()          


#plot natural and gray coding in one plot for window size 1.66
plt.title("Natural and Gray Coding Error Rate for Window Size 0.555 Sec")

plt.semilogy(b[hhh]["naive"][1,:]/3,b[hhh]["naive"][2,:], "-o", label="Naive Approach - NC")
plt.semilogy(b[hhh]["multi_output_model"][1,:]/3,b[hhh]["multi_output_model"][2,:], "-o", label="Machine Learning Model - NC")
plt.semilogy(b[hhh]["naive"][1,:]/3,b[hhh]["naive"][3,:], "-o", label="Naive Approach - GC")
plt.semilogy(b[hhh]["multi_output_model"][1,:]/3,b[hhh]["multi_output_model"][3,:], "-o", label="Machine Learning Model - GC")  
plt.xlabel("Molecule Number per Bit")
plt.ylabel("Bit Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("n_gc_error_window_bit_0555.jpg")    
plt.clf()    

#plot natural and gray coding in one plot for window size 0.5
plt.title("Natural and Gray Coding Bit Error Rate for Window Size 0.166 Sec")

plt.semilogy(b["0.5"]["naive"][1,:]/3,b["0.5"]["naive"][2,:], "-o", label="Naive Approach - NC")
plt.semilogy(b["0.5"]["multi_output_model"][1,:]/3,b["0.5"]["multi_output_model"][2,:], "-o", label="Machine Learning Model - NC")
plt.semilogy(b["0.5"]["naive"][1,:]/3,b["0.5"]["naive"][3,:], "-o", label="Naive Approach - GC")
plt.semilogy(b["0.5"]["multi_output_model"][1,:]/3,b["0.5"]["multi_output_model"][3,:], "-o", label="Machine Learning Model - GC")   
plt.xlabel("Molecule Number per Bit")
plt.ylabel("Bit Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("n_gc_error_window_bit_0166.jpg")    
plt.clf()       

#plot natural and gray coding in one plot for constant molecule size and varying Tb
plt.title("Natural and Gray Coding Bit Error Rate for Molecule Size 750")

plt.semilogy(tb_naive[:,0]/3,tb_naive[:,2], "-o", label="Naive Approach - NC")
plt.semilogy(tb_naive[:,0]/3,tb_multi[:,2], "-o", label="Machine Learning Model - NC")
plt.semilogy(tb_naive[:,0]/3,tb_naive[:,3], "-o", label="Naive Approach - GC")
plt.semilogy(tb_naive[:,0]/3,tb_multi[:,3], "-o", label="Machine Learning Model - GC")   
plt.xlabel("Transmission time per bit (Sec)")
plt.ylabel("Bit Error Rate")
plt.legend(loc=3)
plt.grid()
plt.savefig("n_gc_error_window_tb.jpg")    
plt.clf()


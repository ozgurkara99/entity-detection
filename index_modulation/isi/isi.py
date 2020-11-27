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
downsample = 0.1
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

def preprocess2(size, time, downsample, tri):
    output = np.zeros((len(tri),8 + 1,int(size)))
    print(output.shape)
    classes = np.zeros((len(tri),1))
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
        classes[x,0] = lis[0,-1] 
    return classes, output

def model_train(data, classes):
    keras.backend.clear_session()
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=test_size)
    
    inputs = Input(shape=(X_train.shape[1],X_train.shape[2],1))
#    x = Conv2D(1024, kernel_size=(8,5),strides=(1,5), activation="relu")(inputs)
    x = Conv2D(512, kernel_size = 2)(inputs)
    x = Conv2D(512, kernel_size = 2)(x)
    x = Conv2D(512, kernel_size = 2)(x)
    x = Conv2D(512, kernel_size = 2)(x)
    x = Flatten()(inputs)
    x = Dense(512, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    classes_dict_train = {}
    classes_dict_test = {}
    output_layer = {}
    loss_dic = {}
    for i in range(1,numberr+1):
        classes_dict_train["y_train_" + str(i)] = to_categorical(y_train[:,i-1] - 1)
        classes_dict_test["y_test_" + str(i)] = to_categorical(y_test[:,i-1] - 1)
        output_layer["output_" + str(i)] = Dense(8, activation="softmax")(x)
        loss_dic["dense_" + str(i+2)] = 'categorical_crossentropy'
    model = Model(inputs=inputs, outputs=list(output_layer.values()))
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
    epochs=200
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    #sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(optimizer=opt, loss=loss_dic, metrics=['accuracy'])
    history = model.fit(X_train, list(classes_dict_train.values()),
                        validation_data=(X_test, list(classes_dict_test.values())), epochs=epochs, batch_size=128)
    return history, model    

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

def plot_learning_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    plt.show()   
    
def plot_learning_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'val_accuracy'], loc='upper left')
    plt.show()        
    
    
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

def model_train_masked(data, classes):
    keras.backend.clear_session()
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.01)
    y_train = to_categorical(y_train - 1)
    y_test = to_categorical(y_test - 1)
    inputs = Input(shape=(X_train.shape[1],X_train.shape[2],1))
#    x = Conv2D(128, kernel_size=(8,5))(inputs)
    x = Flatten()(inputs)
    x = Dense(256, activation="relu")(x)   
    x = Dense(1024, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    y = Dense(8, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=y)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9)
    epochs=200
    opt = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test), epochs=epochs, batch_size=256)
    
    return history, model    



def test(test_path, model):
    onlyfiles = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    symbol_rate = np.zeros((2,len(onlyfiles)))
    for i,file in enumerate(onlyfiles):
        test_mol_num = int(between(file,"_","."))
        symbol_rate[1,i] = test_mol_num
        tri = read_data(test_path + file)
        classes, data = preprocess2(size, time, downsample, tri) 
        print(data.shape, classes.shape)
        data_reshaped = data[:,:-1,:]
        data_reshaped = data_reshaped / test_mol_num
        data_reshaped = data_reshaped.reshape(data_reshaped.shape[0],data_reshaped.shape[1],data_reshaped.shape[2],1)
        true_data,true_classes = change_data_and_class_to_masked(data_reshaped, classes)
        predicted_classes = model.predict(true_data)
        pred_classes = np.argmax(predicted_classes, axis=1) + 1
        break
    return pred_classes, true_classes    
    


def select_data(number, data_reshaped, classes):
    new_data = data_reshaped[:,:,:(number+1)*window,:]
    new_classes = classes[:,number]
    return new_data, new_classes   
"""    
tri = read_new(folder_path)
data = {}
for i in range(1,9):
    data[str(i)] = []
for lis in tri:
    data[str(lis[0,-1])[0]].append(lis)
  
import os 
for numberr in range(9,10):
    window_size = time / numberr 
    window = int(window_size*10)
    upper_limit = min(4,numberr)
    
    data_folder = "window_" + str(window_size) + "_upper_" + str(upper_limit) + "\\"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    for hhh in range(2,5):
        isi_data = []
        isi_order = []
        train_size = 2000
        classes = np.zeros((train_size,int(numberr)))
        for j in range(0,train_size):
            z = np.zeros((1,int(numberr)))
            if(j%100==0):
                print("j equals to: " + str(j) + "/" + str(train_size))
            for i in range(0,int(numberr)):
                rand = random.randint(1,8)
                z[0,i] = rand
                rand2 = random.randint(0,149)
                x = data[str(rand)][rand2]
                if(i == 0):
                    deneme = np.squeeze(x[:,np.where((x[2,:-1] + ((i)*window_size) < (window_size*(i+upper_limit))) & (x[2,:-1] + ((i)*window_size) < (time)))])
                    isi_data.append(np.squeeze(x[:,np.where((x[2,:-1] + ((i)*window_size) < (window_size*(i+upper_limit))) & (x[2,:-1] + ((i)*window_size) < (time)))]))
                    
                else:
                    length = len(np.where((x[2,:-1] + ((i)*window_size) < (window_size*(i+upper_limit))) & (x[2,:-1] + ((i)*window_size) < (time)))[0])
                    temp = np.zeros((3,length))
                    row = np.ones((1,length)) * ((i)*window_size)
                    temp[2,:] = row
                    y = np.squeeze(x[:,np.where((x[2,:-1] + ((i)*window_size) < (window_size*(i+upper_limit))) & (x[2,:-1] + ((i)*window_size) < (time)))]) + temp
                    print(y.shape, i)
                    #y = y.reshape(y.shape[0], y.shape[2])
                    isi_data[j] = np.hstack((isi_data[j], y))
            isi_data[j] = isi_data[j][ :, isi_data[j][2].argsort()]
            classes[j,:] = z
    #        with open('isi_data.txt', 'a') as f:
    #            np.savetxt(f,isi_data[j], delimiter=",", fmt="%.4f")
    #        isi_order.append(y)
    #        with open('isi_y.txt', 'a') as f:
    #            np.savetxt(f,z, delimiter=",", fmt="%.1f")
        
        #tri = read_data("isi_data.txt") 
        tri = isi_data
        _, datax = preprocess2(size, time, downsample, tri) 
        data_reshaped = datax[:,:-1,:]
        data_reshaped = data_reshaped / mol_num
        data_reshaped = data_reshaped.reshape(data_reshaped.shape[0],data_reshaped.shape[1],data_reshaped.shape[2],1)
        #hist, model = model_train(data_reshaped, classes) 
        np.save(data_folder + "data_reshaped_" + str(hhh) + ".npy", data_reshaped)
        np.save(data_folder + "classes_reshaped_" + str(hhh) + ".npy", classes)


""" 
import os 
for numberr in range(2,10):
    numberr = 5
    window_size = time / numberr 
    window = int(window_size*10)
    upper_limit = min(4,numberr)      
    data_folder = "window_" + str(window_size) + "_upper_" + str(upper_limit) + "\\"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    data_dict = {}
    shap = [0]
    for i in range(0,5):
        data_dict["x_" + str(i)] = np.load(data_folder + "data_reshaped_" + str(i) + ".npy")
        shap.append(data_dict["x_" + str(i)].shape[0] + shap[i])
        print(data_dict["x_" + str(i)].shape[0], shap[i], shap, i)
        data_dict["y_" + str(i)] = np.load(data_folder + "classes_reshaped_" + str(i) + ".npy")  
    
    data_reshaped = np.zeros((shap[-1], 8, 50, 1))
    classes = np.zeros((shap[-1], numberr))
    for i in range(0,5):
        data_reshaped[shap[i]:shap[i+1],:,:,:] = data_dict["x_" + str(i)]
        classes[shap[i]:shap[i+1], :] = data_dict["y_" + str(i)]
          
    models_dict = {}
    history_dict = {}
    for i in range(numberr):
        new_data_reshaped, new_classes = select_data(i, data_reshaped, classes)
        #keras.backend.clear_session()
        X_train, X_test, y_train, y_test = train_test_split(new_data_reshaped, new_classes, test_size=0.2)
        y_train = to_categorical(y_train - 1)
        y_test = to_categorical(y_test - 1)
        inputs = Input(shape=(X_train.shape[1],X_train.shape[2],1))
        x = Conv2D(128, kernel_size=2)(inputs)
        x = Conv2D(128, kernel_size=2)(x)
        x = Flatten()(inputs)
        x = Dense(256, activation="relu")(x)   
        y = Dense(8, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=y)
        epochs=200
        opt = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test), epochs=epochs, batch_size=256)
        models_dict["model_" + str(i+1)] = model
        model.save(data_folder + "multi_model_" + str(i) + ".h5")
        history_dict["history_" + str(i+1)] = history
       
    hist, model = model_train(data_reshaped, classes)
    model.save(data_folder + "multi_output_model.h5")
    a,b = change_data_and_class_to_masked(data_reshaped, classes)
    hist, model = model_train_masked(a, b)
    model.save(data_folder + "data_masked.h5")


x = np.arange(0.1,5.1,0.1)
y = np.squeeze(data_reshaped)
#y = y * mol_num
for i in range(1,9):   
    if i == 1:
        plt.title("Multivariate Time Series Input (1-7-4-7-1)")
    ax = plt.subplot(8, 1, i)
    plt.plot(x, y[3,i-1,:])
    plt.yticks([0,0.005])
    if not i==8:
        plt.xticks([])      
    if i==5:
        ax.yaxis.set_label_coords(-0.12,1.2)
        plt.ylabel("Emitted Molecule Rate", fontsize=16)
#plt.xlabel("Each Subplot Shows the Percentage of Emitted Molecules From Each Receiver")
plt.xlabel("Time", fontsize=16)

plt.show()
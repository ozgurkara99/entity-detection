import numpy as np
import csv
import math
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tensorflow import keras
from os import listdir
from os.path import isfile, join

PI = math.pi
time = 1
downsample = 0.01
az_pair, el_pair = 4, 4
test_size = 0.4
filepath = 'data\\output_training.csv'
size = time / downsample
mol_num = 100000
"""
1 -> theta yani azimuth
2 -> phi yani elevation
"""

def az_el_pair(az_num, el_num):
    azimuth = np.linspace(-PI, PI, az_num + 1)
    elevation = np.linspace(0,PI, el_num + 1)
    return azimuth, elevation

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


def preprocess(az_pair, el_pair, size, time, downsample, filepath):
    tri = read_data(filepath)
    output = np.zeros((len(tri),az_pair * el_pair + 1,int(size)))
    output[:,az_pair * el_pair,:] = np.linspace(0,time - downsample,int(size))
    classes = np.zeros((len(tri),1))
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
                            classes[inn,0] = lis[0,-1] 
    return classes, output

def train(data, classes, test_size):
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=test_size)
    y_train = to_categorical(y_train)[:,1:]
    y_test = to_categorical(y_test)[:,1:]
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=2, activation="relu", input_shape=(X_train.shape[1],X_train.shape[2],1)))
    model.add(Conv2D(32, kernel_size=2, activation="relu",))
    model.add(Flatten())
    model.add(Dense(8, activation="softmax"))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=800)
    return history, model

def plot_learning_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.show()    
    
def plot_accuracy(history):
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])    
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend([ 'train_acc', 'val_acc'], loc='upper right')
    plt.show()
    
def test(flpath, model, mol_number):
    classes, output = preprocess(az_pair, el_pair, size, time, downsample, filepath = flpath)
    data = output[:,:az_pair * el_pair,:] 
    data = data / mol_number
    data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
    result = model.predict(data[:,:,:,:])
    return classes, ((result.argmax(axis=1)) + 1).reshape((result.shape[0],1))


def symbol(y_pred, y_real):
    length = y_pred.shape[0]
    true_num = np.count_nonzero(y_pred == y_real)
    return 1 - true_num / length

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

def find_mol_rate(filepath, model):
    onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    symbol_rate = np.zeros((2,len(onlyfiles)))
    for i,file in enumerate(onlyfiles):
        test_mol_num = int(between(file,"_","."))
        symbol_rate[1,i] = test_mol_num
        test_class, test_prediction = test(filepath + file, model, test_mol_num)
        symbol_rate[0,i] = symbol(test_prediction, test_class)
        print(str(symbol_rate[0,i]) + " " + str(i+1) + "/" + str(len(onlyfiles)) + " " + str(test_mol_num))
    return symbol_rate
        
def plot_error(data, logarithmic=True):
    plt.title("error - molecule number")
    if(logarithmic):    
        plt.semilogy(data[1,:], data[0,:])
    else:
        plt.scatter(data[1,:],data[0,:])
    plt.xlabel("molecule number")
    plt.ylabel("error")

"""    for training
classes, output = preprocess(az_pair, el_pair, size, time, downsample, filepath)
data = output[:,:az_pair * el_pair,:] 
data = data / mol_num
data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
history, model = train(data,classes, test_size)
plot_learning_curve(history)
plot_accuracy(history)


model.save('cnn_model_normalized.h5')
y_sonuc = model.predict(data[:,:,:,:])
"""

model = keras.models.load_model('cnn_model_normalized.h5')
symbol2 = find_mol_rate("data\\data 0.001\\", model)
"""
symbol3 = find_mol_rate("data\\temp10\\", model)
symbol2 = np.concatenate((symbol2, symbol3), axis=1)
test_mol_number = 3380
test_class, test_prediction = test("data\\temp9\\output_" + str(test_mol_number) + ".csv", model, test_mol_number)
print(symbol(test_prediction, test_class))
"""
plot_error(symbol2)
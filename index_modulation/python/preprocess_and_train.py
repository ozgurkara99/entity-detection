import numpy as np
import csv
import math
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tensorflow import keras

PI = math.pi
time = 0.75
downsample = 0.01
az_pair, el_pair = 4, 4
test_size = 0.2
filepath = 'data\\output.csv'
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
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=700)
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

classes, output = preprocess(az_pair, el_pair, size, time, downsample, filepath)
data = output[:,:az_pair * el_pair,:] 
data = data / mol_num
data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
history, model = train(data,classes, test_size)
plot_learning_curve(history)
plot_accuracy(history)


model.save('cnn_model_normalized.h5')
y_sonuc = model.predict(data[:,:,:,:])



test_class, test_prediction = test('data\\output5.csv', model, 5000)
print(symbol(test_prediction, test_class))

    
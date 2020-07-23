import numpy as np
import csv
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tensorflow import keras
from os import listdir
from os.path import isfile, join

PI = np.pi
pi = np.pi
time = 1
downsample = 1
az_pair, el_pair = 4, 4
test_size = 0.4
filepath = 'data\\training\\output_100000.csv'
size = time / downsample
mol_num = 100000
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

def max_preprocess(filepath, offset=0):
    onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    symbol_rate = np.zeros((2,len(onlyfiles)))
    for i,file in enumerate(onlyfiles):
        true = 0
        test_mol_num = int(between(file,"_","."))
        symbol_rate[1,i] = test_mol_num
        data = read_data(filepath + file)
        print("processing: " + filepath + file)
        sample_size = len(data)
        tri = np.zeros((8,8,sample_size))
        for x,lis in enumerate(data):
            for time in range(len(lis[0]) - 1):
                if(lis[2,time] >= 1 - offset):
                    break
                j = int(lis[0,-1]) - 1
                _, y, z = sph2cart(lis[0,time], lis[1,time], 5)
                aci = np.arctan2(y,z)
                if(aci < pi/8 and aci >= -pi/8):
                    tri[j,2,x] = tri[j,2,x] + 1
                elif (aci >= pi/8 and aci < 3*pi/8):
                    tri[j,1,x] = tri[j,1,x] + 1
                elif (aci >= 3*pi/8 and aci < 5*pi/8):
                    tri[j,0,x] = tri[j,0,x] + 1
                elif (aci >= 5*pi/8 and aci < 7*pi/8):
                    tri[j,7,x] = tri[j,7,x] + 1
                elif (aci >= 7*pi/8 and aci <= pi) or ( aci < -7*pi/8):
                    tri[j,6,x] = tri[j,6,x] + 1
                elif (aci >= -7*pi/8 and aci < -5*pi/8):
                    tri[j,5,x] = tri[j,5,x] + 1 
                elif (aci >= -5*pi/8 and aci < -3*pi/8):
                    tri[j,4,x] = tri[j,4,x] + 1
                elif (aci >= -3*pi/8 and aci < -pi/8):
                    tri[j,3,x] = tri[j,3,x] + 1                                                                                 
        for k in range(sample_size):
            if(np.argmax(np.max(tri[:,:,k], axis=1)) == np.argmax(np.max(tri[:,:,k], axis=0))):
                true += 1
        symbol_rate[0,i] = (sample_size - true) / (sample_size)
    return symbol_rate
        
for i in (np.arange(0,0.50,0.05)):   
    symbol = max_preprocess(filepath,i)
    np.savetxt("txtfiles2\\offset_" + str(i), symbol)
    
symbol = []
for x,i in enumerate(np.arange(0,0.50,0.05)):   
    a = np.loadtxt("txtfiles2\\offset_" + str(i))
    plot_error(a, label=f'{(1-i):.2f}' + "s")


       
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

def preprocess2(az_pair, el_pair,size, time, downsample, filepath):
    tri = read_data(filepath)
    output = np.zeros((len(tri),8 + 1,int(size)))
    output[:,8,:] = np.linspace(0,time - downsample,int(size))
    classes = np.zeros((len(tri),1))
    for x,lis in enumerate(tri):
        i = 0
        for timex in range(int(size)):
            while((timex * downsample) <= lis[2,i] and (lis[2,i] < (timex + 1) * downsample)):
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

def train2(data, classes, test_size):
    data = data.reshape(data.shape[0],data.shape[1])
    data = data / mol_num
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=test_size)
    y_train = to_categorical(y_train)[:,1:]
    y_test = to_categorical(y_test)[:,1:]
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model = Sequential()
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
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
    classes, output = preprocess2(az_pair, el_pair, size, time, downsample, filepath = flpath)
    #data = output[:,:az_pair * el_pair,:] 
    data = output[:,:8,:] 
    data = data / mol_number
    data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
    result = model.predict(data[:,:,:,:])
    return classes, ((result.argmax(axis=1)) + 1).reshape((result.shape[0],1))


def symbol(y_pred, y_real):
    length = y_pred.shape[0]
    true_num = np.count_nonzero(y_pred == y_real)
    return 1 - true_num / length

def test2(flpath, model, mol_number):
    classes, output = preprocess2(az_pair, el_pair, size, time, downsample, filepath = flpath)
    data = output[:,:8,:] 
    #data = output[:,:az_pair * el_pair,:] 
    data = data / mol_number
    data = data.reshape(data.shape[0],data.shape[1])
    result = model.predict(data)
    return classes, ((result.argmax(axis=1)) + 1).reshape((result.shape[0],1))

def find_mol_rate(filepath, model):
    onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    symbol_rate = np.zeros((2,len(onlyfiles)))
    for i,file in enumerate(onlyfiles):
        test_mol_num = int(between(file,"_","."))
        symbol_rate[1,i] = test_mol_num
        test_class, test_prediction = test2(filepath + file, model, test_mol_num)
        symbol_rate[0,i] = symbol(test_prediction, test_class)
        print(str(symbol_rate[0,i]) + " " + str(i+1) + "/" + str(len(onlyfiles)) + " " + str(test_mol_num))
    return symbol_rate
        
def plot_error(data, label, logarithmic=True):
    plt.title("Error - Molecule number")
    if(logarithmic):    
        plt.semilogy(data[1,:], data[0,:], "-o", label=label)
    else:
        plt.scatter(data[1,:],data[0,:], "-o", label=label)
    plt.xlabel("Molecule Number")
    plt.ylabel("Error")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

"""    for training
classes, output = preprocess2(az_pair, el_pair, size, time, downsample, filepath)
data = output[:,:az_pair * el_pair,:] 
data = output[:,:8,:] 
data = data / mol_num
data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
history, model = train(data,classes, test_size)
history, model = train2(data,classes, test_size)
plot_learning_curve(history)
plot_accuracy(history)


model.save('karpuz_downsampled_to_1.h5')
y_sonuc = model.predict(data[:,:,:,:])
"""

model = keras.models.load_model('karpuz_downsampled_to_0.01.h5')
symbol2 = find_mol_rate("data\\500\\", model)
np.savetxt("error_texts_karpuz\\downsampled_to_1_.txt", symbol2)
"""
symbol3 = find_mol_rate("data\\temp10\\", model)
symbol2 = np.concatenate((symbol2, symbol3), axis=1)
test_mol_number = 3380
test_class, test_prediction = test("data\\temp9\\output_" + str(test_mol_number) + ".csv", model, test_mol_number)
print(symbol(test_prediction, test_class))
"""
plot_error(symbol2, label="downsampled-to-1")

symbola = np.loadtxt("error_texts_karpuz\\downsampled_to_1_.txt")
symbolb = np.loadtxt("error_texts_karpuz\\downsampled_to_0.1_.txt")
symbolc = np.loadtxt("error_texts_karpuz\\downsampled_to_0.01_.txt")
plot_error(symbola, label="Karpuz - 1 second")
plot_error(symbolb, label="Karpuz - 0.1 second")
plot_error(symbolc, label="Karpuz - 0.01 second")
symbola = np.loadtxt("error_texts\\downsampled_to_1_.txt")
symbolb = np.loadtxt("error_texts\\downsampled_to_0.1_.txt")
symbolc = np.loadtxt("error_texts\\downsampled_to_0.01_.txt")
plot_error(symbola, label="1 second")
plot_error(symbolb, label="0.1 second")
plot_error(symbolc, label="0.01 second")
symbola = np.loadtxt("txtfiles2\\offset_0.0")
plot_error(symbola, label="Looking maximum molecule region")
        
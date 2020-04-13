import pandas as pd
from keras.layers import Dense, Activation, BatchNormalization, Input
from keras.layers import * 
from keras.models import Model
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras import optimizers
import keras.backend as K
import keras
import os
from matplotlib import pyplot as plt

folder_path = "data/"

def readData(folder_path):
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for id,file in enumerate(onlyfiles):
        if(id == 0):
            newdata = np.loadtxt(folder_path+file,delimiter=',',skiprows=0)
        else:
            data = np.loadtxt(folder_path+file,delimiter=',',skiprows=0)
            newdata = np.concatenate((newdata, data), axis=0)
    return newdata

newdata = readData(folder_path)
y = newdata[:,1000:]
y[np.where(y[:,0]==0),1:] = 0
newdata = newdata[:,:1000]
X = newdata

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

def MSEloss(prob_true): 
    def loss(coord_true, coord_predict):
        sq = K.sum(tf.multiply(prob_true,tf.square(tf.subtract(coord_true, coord_predict))),axis=1)
        mask = tf.greater(sq, 0)
        non_zero_array = tf.boolean_mask(sq, mask)
        #sq = sq[~np.all(sq == 0, axis=1)]
        mseloss = K.mean(non_zero_array)
        return mseloss
    return loss

def BCEloss():
    def loss(prob_true, prob_predict):
        bceloss = K.mean(keras.losses.binary_crossentropy(prob_true, prob_predict))
        return bceloss
    return loss


s = tf.keras.backend.clear_session()


#Two additional 'inputs' for the labels
prob_layer = Input((1,),name="prob_in")
coords_layer = Input((2,),name="coord_in")

inp = Input((1000,),name="in")
x = BatchNormalization()(inp)

x = Dense(450,activation='relu')(inp)
x = BatchNormalization()(x)
x = Dense(450,activation='relu')(x)
x = BatchNormalization()(x)

x = Dense(450,activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(450,activation='relu')(x)


x = (Dense(450, activation='relu'))(x)

x = (Dense(450, activation='relu'))(x)

out1 = Dense(1, activation='sigmoid', name='probability')(x)
out2 = Dense(2, activation='linear',name='coordinates')(x)
opt = optimizers.Adam(learning_rate=0.01)
sgdd = tf.keras.optimizers.SGD(learning_rate=0.01)
model = Model(inputs=[inp, prob_layer, coords_layer], outputs=[out1, out2])
model.compile(loss=[BCEloss(), MSEloss(prob_layer)], optimizer= optimizers.SGD(lr=0.01, clipvalue=0.7))

history = model.fit(x=[X_train, y_train[:,0], y_train[:,1:]], 
                    y=[y_train[:,0], y_train[:,1:]], 
                    validation_data=([X_val, y_val[:,0], y_val[:,1:]], [y_val[:,0], y_val[:,1:]]),
                    batch_size = 128, 
                    epochs=100)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


prediction = model.predict(x = [X_test,y_test[:,0],y_test[:,1:]])
mse = np.sqrt(np.square(prediction[1][:,0] - y_test[:,1]) + np.square(prediction[1][:,1] - y_test[:,2]))
def round_nearest(x, a):
    return round(x / a) * a
for i in range(len(mse)):
    mse[i] = round_nearest(mse[i], 0.05)
ort = np.mean(mse)
unique, counts = np.unique(mse, return_counts=True)
den = np.stack((unique,counts/np.sum(counts))).T
plt.plot(unique,counts/np.sum(counts))
plt.ylabel('probability')
plt.xlabel('l2 norm of error vectors')

np.savetxt('output.csv',den[:100,:],delimiter=",")
model.save('good_result_0.726_100epoch.h5')
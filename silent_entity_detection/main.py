import pandas as pd
from keras.layers import Dense, Activation, BatchNormalization, Input, Dropout
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
from tensorflow.keras import regularizers

folder_path = "data/"

#read the files in given path and returns an array that contains all the data in these files
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
#extract y
y = newdata[:,1000:]
y[np.where(y[:,0]==0),1:] = 0
#extract X
newdata = newdata[:,:1000]
X = newdata
del newdata
#normalize the data
normalized_x = tf.keras.utils.normalize(
    X, axis=-1, order=2
)
#split the data to train, test, validation set
X_train, X_test, y_train, y_test = train_test_split(normalized_x, y, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

#custom mean squared error loss, when probability is zero, the loss of the coordinates will be ignored
def MSEloss(prob_true): 
    def loss(coord_true, coord_predict):
        sq = K.sum(tf.multiply(prob_true,tf.square(tf.subtract(coord_true, coord_predict))),axis=1)
        mask = tf.greater(sq, 0)
        non_zero_array = tf.boolean_mask(sq, mask)
        #sq = sq[~np.all(sq == 0, axis=1)]
        mseloss = K.mean(non_zero_array)
        return mseloss
    return loss

#binary cross entropy loss
def BCEloss():
    def loss(prob_true, prob_predict):
        bceloss = K.mean(keras.losses.binary_crossentropy(prob_true, prob_predict))
        return bceloss
    return loss

#clear the graph
s = tf.keras.backend.clear_session()

#constants are assigned
BATCH_SIZE = 1024
N_TRAIN = X_train.shape[0]
EPOCH_NUM = 500
STEPS_PER_EPOCH = N_TRAIN/BATCH_SIZE

#make learning rate decreasing as time goes
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*500,
  decay_rate=1,
  staircase=False)

#use Adam optimizer
def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)
optimizer = get_optimizer()

#Two additional 'inputs' for the labels
prob_layer = Input((1,),name="prob_in")
coords_layer = Input((2,),name="coord_in")

#create the model
#elu activation function is used
inp = Input((1000,),name="in")
x = Dense(512, kernel_regularizer=regularizers.l2(0.0001),activation='elu')(inp)
x = Dense(512, kernel_regularizer=regularizers.l2(0.0001),activation='elu')(x)
x = Dense(512, kernel_regularizer=regularizers.l2(0.0001),activation='elu')(x)
x = Dense(512, kernel_regularizer=regularizers.l2(0.0001),activation='elu')(x)
x = Dense(512, kernel_regularizer=regularizers.l2(0.0001),activation='elu')(x)
x = Dense(512, kernel_regularizer=regularizers.l2(0.0001),activation='elu')(x)

#outputs
out1 = Dense(1, activation='sigmoid', name='probability')(x)
out2 = Dense(2, activation='linear',name='coordinates')(x)

#opt = optimizers.Adam(learning_rate=0.01)
#sgdd = tf.keras.optimizers.SGD(learning_rate=0.01)

model = Model(inputs=[inp, prob_layer, coords_layer], outputs=[out1, out2])
model.compile(loss=[BCEloss(), MSEloss(prob_layer)], optimizer= optimizer)

history = model.fit(x=[X_train, y_train[:,0], y_train[:,1:]], 
                    y=[y_train[:,0], y_train[:,1:]], 
                    validation_data=([X_val, y_val[:,0], y_val[:,1:]], [y_val[:,0], y_val[:,1:]]),
                    batch_size = BATCH_SIZE,
                    epochs=EPOCH_NUM)

#plot the loss/epoch graph
plt.plot(history.history['coordinates_loss'][2:])
plt.plot(history.history['val_coordinates_loss'][2:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#predict the test data and get the rounded error vector
prediction = model.predict(x = [X_test,y_test[:,0],y_test[:,1:]])
mse = np.sqrt(np.square(prediction[1][:,0] - y_test[:,1]) + np.square(prediction[1][:,1] - y_test[:,2]))
def round_nearest(x, a):
    return round(x / a) * a
for i in range(len(mse)):
    mse[i] = round_nearest(mse[i], 0.05)
ort = np.mean(mse)
unique, counts = np.unique(mse, return_counts=True)
den = np.stack((unique,counts/np.sum(counts))).T

#plot the error probability/error norm graph
plt.plot(unique,counts/np.sum(counts))
plt.ylabel('probability')
plt.xlabel('l2 norm of error vectors')

x = np.mean(mse[np.where(mse<6)])
np.savetxt('output.csv',den[:,:],delimiter=",")

#save the model
model.save('good_result_0.72_500epoch_enson.h5')
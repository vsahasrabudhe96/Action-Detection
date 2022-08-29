from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from preprocess import train_test_data
import numpy as np
import os


DATA_PATH = "../data/"
actions = np.array(['hello','thank you','I love you']) ## Actions we try to detect
no_sequences = 20  ## no_sequences == number of videos we want to collect e.g here we will collect 30 videos
sequence_length = 20 ## Each video will be of 30 frames

X_train,X_test,y_train,y_test = train_test_data(DATA_PATH,actions,no_sequences,sequence_length)

log_dir = os.path.join("../Logs")
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64,return_sequences = True,activation='relu',input_shape=(20,1662)))
model.add(LSTM(128,return_sequences=True,activation='relu'))
model.add(LSTM(64,return_sequences=False,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(actions.shape[0],activation='softmax'))
    
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.fit(X_train,y_train,epochs=100,callbacks=[tb_callback])
model.save('../models/action.h5')
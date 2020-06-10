import numpy as np
import keras
import sys
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalMaxPool2D, AvgPool2D, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

#train data file read
data = np.load(sys.argv[1])
data_label = np.load(sys.argv[2])
data = np.rollaxis(data,1,4)
data = data.astype(np.float32)
label = keras.utils.to_categorical(data_label, 10)

#data division for cross-validation
length = round(len(data)*0.95)

x_train = data[:length]
y_train = label[:length]
x_valid = data[length:]
y_valid = label[length:]

#print number of training, validation, and test images
print(x_train.shape[0],'train samples')
print(x_valid.shape[0],'valid samples')

#model 
model =Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu',input_shape=(112,112,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu'))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(BatchNormalization(momentum=0.5))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same',activation='relu'))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(BatchNormalization(momentum=0.5))
model.add(Dropout(0.4))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',activation='relu'))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(BatchNormalization(momentum=0.5))
model.add(Dropout(0.4))
model.add(GlobalMaxPool2D())
model.add(Dropout(0.4))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

opt = Adam(lr=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer= opt ,
              metrics=['accuracy'])
			  
#train the model
checkpointer= ModelCheckpoint(filepath=sys.argv[3],verbose=1,save_best_only=True)

model.fit(x_train, y_train,
              batch_size=32,
              epochs=50,
              validation_data=(x_valid, y_valid),
              callbacks=[checkpointer],
              shuffle=True)

#load the weights
model.load_weights(sys.argv[3])	

#save the model		  
model.save(sys.argv[3])

import sys
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


data_file = sys.argv[1]
model_file = sys.argv[2]

num_labels = 101
lr = 0.001
epochs = 30
val_steps = 160
epoch_step = 650



datagen = ImageDataGenerator(validation_split=0.2,rescale=1./255)

train_gen = datagen.flow_from_directory('{}'.format(data_file),
                                    subset='training',
                                    batch_size=64,
                                    target_size=(32,32))

val_gen = datagen.flow_from_directory('{}'.format(data_file),
                                    subset='validation',
                                    batch_size=32,
                                    target_size=(32,32))


inp_tensor = Input(shape=(32, 32, 3))
transfer_model = applications.MobileNet(weights=None, 
                                        input_tensor=inp_tensor,
                                        pooling='avg',
                                        classes=num_labels)
                                        
transfer_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


early_stopping = True

mc = ModelCheckpoint(model_file, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

def train():
    transfer_model.fit_generator(train_gen, 
                        steps_per_epoch=epoch_step, 
                        epochs=epochs,
                        validation_data=val_gen, 
                        validation_steps=val_steps,
                        verbose=2,
                        callbacks=[mc])
train()

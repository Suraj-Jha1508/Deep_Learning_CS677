import sys
import keras
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import preprocess_input
from keras.layers import Conv2D, MaxPooling2D, Input,Flatten, Dense, Dropout, InputLayer,GlobalAveragePooling2D
from keras.models import Sequential
from keras import optimizers
from keras.models import load_model
from keras import regularizers

train_folder=sys.argv[1]
train_data=train_folder

train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
        directory=train_data,
        target_size=(256,256),
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(directory=train_data,
        target_size=(256,256),
        class_mode='categorical',
        subset='validation')

#tensor=Input(shape=(256,256,3))
base_resnet = applications.vgg16.VGG16(weights='imagenet', include_top=False, 
                             input_shape=(256,256,3))
                             
# Add a global spatial average pooling layer
out = base_resnet.output
out = GlobalAveragePooling2D()(out)

out = Dense(128,kernel_initializer='he_normal',activation='relu')(out)
out = Dropout(0.2)(out)
out = Dense(128,kernel_initializer='he_normal',activation='relu')(out)

out = Dropout(0.2)(out)
out = Dense(64,kernel_initializer='he_normal', activation='relu')(out)
out = Dense(32,kernel_initializer='he_normal', activation='relu')(out)

total_classes = 5
predictions = Dense(total_classes, activation='softmax')(out)

model = Model(inputs=base_resnet.input, outputs=predictions)


for layer in base_resnet.layers:
    layer.trainable = False


opt=keras.optimizers.Adam(lr=0.001,beta_1=0.8)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

early_stopping = True
checkpointer=ModelCheckpoint(filepath=sys.argv[2],monitor='val_acc',verbose=1,mode='max',save_best_only=True)

model.fit_generator(
        train_generator,
        steps_per_epoch=200, epochs=20,
        validation_data=validation_generator, validation_steps=80,verbose=2,callbacks=[checkpointer])

model.load_weights(sys.argv[2])
model.save(sys.argv[2])

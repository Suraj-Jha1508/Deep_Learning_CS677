import sys
import keras
import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras import optimizers
from keras.models import load_model

train_data=sys.argv[1]

train_datagen = ImageDataGenerator(
        rescale=1./255,validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
        directory=train_data,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(directory=train_data,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        subset='validation')
		
base_model = InceptionV3(weights='imagenet', include_top=False, 
                             input_shape=(256, 256, 3))
model = base_model.output
model = GlobalAveragePooling2D()(model)
model = Dense(128, activation='relu')(model)
model = Dense(64, activation='relu')(model)
total_classes = 10
predictions = Dense(total_classes, activation='softmax')(model)

model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False


opt=keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


checkpointer=ModelCheckpoint(filepath=sys.argv[2],verbose=1,save_best_only=True)

model.fit_generator(
        train_generator,
        steps_per_epoch=10, epochs=30,
        validation_data=validation_generator, validation_steps=50,callbacks=[checkpointer])

model.load_weights(sys.argv[2])
model.save(sys.argv[2])

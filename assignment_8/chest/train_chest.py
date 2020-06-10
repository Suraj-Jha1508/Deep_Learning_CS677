import sys
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint



train_file = sys.argv[1]
model_file = sys.argv[2]

num_labels = 2
lr = 0.001
epochs = 15
val_steps = 8
epoch_step = 80



datagen = ImageDataGenerator(validation_split=0.2,rescale=1./255)

train_gen = datagen.flow_from_directory('{}'.format(train_file),
                                    target_size=(224,224), batch_size=64)

datagen = ImageDataGenerator(rescale=1./255)
val_gen = datagen.flow_from_directory('{}'.format(train_file),
                                    target_size=(224,224), batch_size=16)


inp_tensor = Input(shape=(224, 224, 3))
transfer_model = applications.InceptionResNetV2(weights='imagenet', 
                                          include_top=False, 
                                          input_tensor=inp_tensor)

for layer in transfer_model.layers:
  layer.trainable = False

out = transfer_model.output
out = Dropout(0.5)(out)
out = GlobalAveragePooling2D()(out)
out = Dense(128, activation='relu')(out)
out = BatchNormalization()(out)
predictions = Dense(num_labels, activation='softmax')(out)

model = Model(inputs=transfer_model.input, outputs=predictions)

opt = optimizers.Adam(lr=0.001, beta_1=0.8)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


early_stopping = True

mc = ModelCheckpoint(model_file, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

def train():
    model.fit_generator(train_gen, 
                        steps_per_epoch=epoch_step, 
                        epochs=epochs,
                        validation_data=val_gen, 
                        validation_steps=val_steps,
                        verbose=2,
                        callbacks=[mc])
train()

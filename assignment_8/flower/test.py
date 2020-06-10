import sys
import keras
import numpy as np
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


test_data=sys.argv[1]

test_datagen = ImageDataGenerator(
        rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        directory=test_data,
        target_size=(256,256),
        batch_size=32,
        class_mode='categorical')

		
#evaluate test accuracy
model = load_model(sys.argv[2])

score = model.evaluate_generator(test_generator,verbose=0)


accuracy=100*score[1]

#print test accuracy after training
print('Test accuracy:',accuracy)

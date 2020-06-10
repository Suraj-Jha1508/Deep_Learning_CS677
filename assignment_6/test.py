import numpy as np
import keras
import sys
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalMaxPool2D, AvgPool2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam

#test data file read
test = np.load(sys.argv[1])
test_label = np.load(sys.argv[2])
test = np.rollaxis(test,1,4)
x_test =  test.astype(np.float32)
y_test = keras.utils.to_categorical(test_label, 10)

model=load_model(sys.argv[3])

#evaluate test accuracy
score=model.evaluate(x_test,y_test,verbose=0)
accuracy=100*score[1]

 #print test accuracy after training
print('Test accuracy:',accuracy)

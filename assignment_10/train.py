import keras
import numpy as np
from keras.models import load_model,save_model
from keras import Input, Sequential
from keras import backend as K
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D
import sys
tf.compat.v1.disable_eager_execution()

def get_indx(num):
  rand_idx = np.random.randint(0, len(traindata),num) 

  while len(set(rand_idx))!= num:
    rand_idx = np.random.randint(0, len(traindata),num) 
  return((rand_idx))
  
def predicted_labels(dataset_D,model_M):

  labels_D = model_M.predict(dataset_D)
  labels_D = np.array(list(map(np.argmax, labels_D)))
  labels_D = keras.utils.to_categorical(labels_D, 2)

  return labels_D

# model -B 
def model_B():
  model_B = Sequential()

  model_B.add(Conv2D(1,kernel_size=(3,3),activation= 'relu',padding= 'same',input_shape=(32,32,3)))
  model_B.add(Flatten())
  model_B.add(Dense(100,activation='relu'))
  model_B.add(Dense(100,activation='relu'))
  model_B.add(Dense(2,activation='softmax'))
  
  
  model_B.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

  return model_B

def cal_adversis_epsilon(data,gradient,epsilon):
  adversaries=data + epsilon * np.sign(gradient)
  return adversaries
  
def cal_adversis(data,gradient):
  sign_V=np.random.uniform(-1,1,len(data))
  sign_V= 0.1* np.sign(sign_V)

  adversaries=data + sign_V[:,None,None,None] * np.sign(gradient)

  return adversaries

def get_gradient(data, model):
  tf.compat.v1.disable_eager_execution()
  gradients = K.gradients(model.output, model.input)[0]
  sess = tf.compat.v1.InteractiveSession()
  sess.run(tf.compat.v1.global_variables_initializer())
  eval_gradients = sess.run(gradients, feed_dict={model.input:data})

  return eval_gradients
'''
def get_gradient(data, model):
  gradients = K.gradients(model.output,model.input)[0]
  iterate = K.function(model.input, gradients)
  grad = iterate([data])

  return grad[0]
'''
def get_test_minus(data, labels, idxs):
  new_idxs = np.arange(data.shape[0])
  idxs_mask = np.isin(new_idxs, idxs, invert=True)
  new_idxs = new_idxs[idxs_mask]

  test_minus_data = data[new_idxs]
  test_minus_labels = labels[new_idxs]
  
  return test_minus_data, test_minus_labels

#read data


traindata = np.load(sys.argv[1])
trainlabels = np.load('test_label.npy')

traindata = traindata[np.logical_or(trainlabels == 0, trainlabels == 1)]
trainlabels = trainlabels[np.logical_or(trainlabels == 0, trainlabels == 1)]
trainlabels = keras.utils.to_categorical(trainlabels, 2)

trainDataMean = np.mean(traindata, axis=0)
traindata = traindata - trainDataMean

indx = get_indx(200)
dataset_D = traindata[indx]

#load model_M (targeted model)
model_M = load_model(sys.argv[2])

# create model_B (substitue model)
model_B = model_B()

# test data
test_minus_data, test_minus_labels = get_test_minus(traindata, trainlabels, indx)

np.save('test_data.npy',test_minus_data)
np.save('test_labels.npy',test_minus_labels)

# predicted label by target model 
predicted_test_minus_labels = predicted_labels(test_minus_data,model_M)

orignal_acc = accuracy_score(test_minus_labels,predicted_test_minus_labels)
print('Orignal_accuracy: {}'.format(orignal_acc))

for i in range(10):
  # trainging model_B
  labels_D = predicted_labels(dataset_D,model_M)

  model_B.fit(dataset_D, labels_D,epochs=10,batch_size=32,verbose=0)

  gradient = get_gradient(dataset_D, model_B)
  adv  = cal_adversis(dataset_D, gradient)

  dataset_D = np.append(adv,dataset_D,axis=0)
  
  # testing model_B performance
  gradient = get_gradient(test_minus_data, model_B)
  test_adv = cal_adversis_epsilon(test_minus_data, gradient, 0.0625)

  predicted_new_test_labels = predicted_labels(test_adv,model_M)

  acc = accuracy_score(test_minus_labels,predicted_new_test_labels)
  print('epoch:',i)
  print('accuracy: {}'.format(acc))

save_model(model_B,sys.argv[3])

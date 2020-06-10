from keras.models import load_model
import keras
import numpy as np
from keras import backend as K
from sklearn.metrics import accuracy_score
import sys
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def cal_adversis_epsilon(data,gradient,epsilon):
  adversaries=data + epsilon * np.sign(gradient)
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
def predicted_labels(data,model):

  labels_D = model.predict(data)
  labels_D = np.array(list(map(np.argmax, labels_D)))
  labels_D = keras.utils.to_categorical(labels_D, 2)

  return labels_D
test_data_file = sys.argv[1]
target_model_file = sys.argv[2]
bb_model_file = sys.argv[3]

test_label_file = 'test_labels.npy'
test_data = np.load(test_data_file)
test_labels = np.load(test_label_file)


m_model = load_model(target_model_file)

b_model = load_model(bb_model_file)

predicted_test_minus_labels = predicted_labels(test_data,m_model)

orignal_acc = accuracy_score(test_labels,predicted_test_minus_labels)
print('Orignal_accuracy: {}'.format(orignal_acc))

gradient = get_gradient(test_data, b_model)
test_adv = cal_adversis_epsilon(test_data, gradient, 0.0625)

predicted_new_test_labels = predicted_labels(test_adv,m_model)

final_acc = accuracy_score(test_labels,predicted_new_test_labels)
print('final accuracy: {}'.format(final_acc))

print("Accuracy Dropped a Total By :", orignal_acc - final_acc)

import keras
import sys
from keras.layers import Conv2D,Input, MaxPooling2D, Flatten, Dense, Dropout, InputLayer,GlobalAveragePooling2D,BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.models import load_model

Model = load_model(sys.argv[1])
    


def test_generated_images(generator, examples=100, dim=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,28,28)
    plt.figure(figsize=(10,10))
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('final_image.png')


test_generated_images(Model, examples=100, dim=(10,10))

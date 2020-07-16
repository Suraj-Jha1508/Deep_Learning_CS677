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


train_data=sys.argv[1]
#model_file=sys.argv[1]

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)
(X_train, y_train,X_test, y_test)=load_data()
print(X_train.shape)

def adam_optimizer():
    return Adam(lr=0.0001,beta_1=0.5)

def generator_model():
    generator=Sequential()
    generator.add(Dense(units=256,input_dim=100))
    generator.add(BatchNormalization()) 
    generator.add(LeakyReLU(0))
    
    generator.add(Dense(units=512))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=1024))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=784, activation='tanh'))
    
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer(),metrics=['accuracy'])
    return generator

gen=generator_model()



def discriminator_model():
    discriminator=Sequential()
    discriminator.add(Dense(units=1024,input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer(),metrics=['accuracy'])
    return discriminator
disc =discriminator_model()



def create_gan_network(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    return gan

gan = create_gan_network(disc,gen)
gan.summary()

def save(x):
    model=save_model(x,model_file)
    return model	


def plotting_generated_images(epoch, generator, examples=100, dim=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,28,28)
    plt.figure(figsize=(10,10))
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('geneartive_model_image_{:04d}.png'.format(epoch))


def training(epochs=1, batch_size=128):
    (X_train, y_train, X_test, y_test) = load_data()
    
    generator= generator_model()
    discriminator= discriminator_model()
    gan = create_gan_network(discriminator, generator)
   
    for epoch in range(1,epochs + 1 ):
        print("Epoch {}/{}".format(epoch,epochs))
        for _ in range(batch_size):
            noise= np.random.normal(0,1, [batch_size, 100])
            
            generated_images = generator.predict(noise)
            
            image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
             
            X= np.concatenate([image_batch, generated_images])
            
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9
           
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            
            noise= np.random.normal(0,1, [batch_size, 100])
            y_gen = np.ones(batch_size)
            
            discriminator.trainable=False
            
            gan.train_on_batch(noise, y_gen)
            
        if epoch == 1 or epoch % 20 == 0:
            plotting_generated_images(epoch, generator)
    generator.save(sys.argv[2])

training(400,128)



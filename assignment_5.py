# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 01:04:03 2020

@author: Suraj
"""
import numpy as np
import sys

class firstCNN:
    
    def __init__(self,train_data,label,test_data,test_label,dimension,stride,eta):
        '''
        
        Function: Constructor of class
        
        '''
        self.train=train_data
        self.label=label
        self.test=test_data
        self.test_label=test_label
        self.x,self.y=dimension
#        self.c_value=np.array([[1,-1],[-1,1]]).flatten()
        self.c_value=np.random.rand(self.x,self.y).flatten()
        self.data_count=len(self.train)
        self.rows=len(self.train[0])
        self.cols=len(self.train[0][0])
        self.stride=stride
        self.eta=eta
        
    def dataTransformation(self,data):
        '''
        
        Function: transforming Train data into filter matrix shape
                  for easier calculation
        
        '''
        x_0=int(self.cols-self.x/self.stride)+1
        y_0=int(self.rows-self.y/self.stride)+1
        
        x=np.empty((len(data), (self.x+self.y), (x_0*y_0)))
        print(x.shape)
        for k in range(len(data)):
            data_temp=[]
            for i in range(x_0):
                for j in range(y_0):
                    data_temp.append(data[k][i:i+self.x,j:j+self.y].flatten())
            x[k]=np.array(data_temp)
        
        if (data.all() == self.train.all()):
            print('update')
            self.train = x
        elif(data.all() == self.test.all()):
            print('done')
            self.test = x
          
    def z_value(self):
        '''
        
        Function: covolution result calculation function
        
        '''
        z_value=np.multiply(self.train,self.c_value[:,np.newaxis])
        z_value=z_value.sum(axis=1)
        z_value=self.sigmoid(z_value)
        
        return(z_value)
    
    def sigmoid(self,x):
        '''
        
        Function: Sigmoid function
        
        '''
        return (1/(1+np.exp(-x)))
    
    def get_fltr(self):
        '''
        
        Function: Filter function
        
        '''
        new = self.c_value
        
        new[np.where(new> np.mean(self.c_value))] = 1
        new[np.where(new < np.mean(self.c_value))] = -1
        
        return new
    
    def model_train(self):
        '''
        
        Function: trainning function
        
        '''
        print(self.train.shape)
        self.dataTransformation(self.train)
        print(self.train.shape)
        prev_obj=np.inf
        epochs=10000
        iteration=1
        while (iteration<epochs):
            
            #dellz calulation
            z_value=self.z_value()
            z_prime=z_value*(1-z_value)

            f=(z_value.mean(axis=1)-self.label)
            
            
            dellz=np.multiply(self.train, z_prime[:,np.newaxis])
            dellz=dellz.sum(axis=2)
            dellz=np.multiply(f[:,np.newaxis],dellz).sum(axis=0)/4
            
            #c update
            self.c_value=np.subtract(self.c_value,(dellz*self.eta))

            
            #objective calculation
            z_value=z_value=self.z_value()
            obj=(np.square(z_value.mean(axis=1)-self.label)).sum(axis=0)
            print('obj:\t{}\t,epochs:\t{} '.format(obj,iteration))
            
            
            if prev_obj-obj<0.0001:
                break
            prev_obj=obj
            iteration+=1
            
            
        self.get_fltr()
        print('Value of C:\n',self.c_value.reshape(2,2))
           
    
    def accuracy_test(self):
        '''
        
        Function: Accuracy testing  function
        
        '''
        print(self.test.shape)
        self.dataTransformation(self.test)
        print(self.test.shape)
        z_value=np.multiply(self.train,self.c_value[:,np.newaxis])
        z_value=z_value.sum(axis=1)
        z_value=self.sigmoid(z_value)
        output=z_value.mean(axis=1)
        print(output)
        new=output
        new[np.where(new> np.mean(output))] = 1
        new[np.where(new < np.mean(output))] = 0
        print('Output: ',new)
        count=0
        for i in range(len(self.test_label)):
            if new[i]==self.test_label[i]:
                count+=1
        print('Accuracy:{}%'.format((count/len(self.test_label))*100))
        
if __name__=='__main__':
    
    data = open('{}/data.csv'.format(sys.argv[1]))
    data=data.readlines()
    data = np.array([data[i].strip().split(',') for i in range(1,len(data))])
    train_img_file = data[:, 0]
    train_label = data[:, 1].astype(np.float32)
    
    data = open('{}/data.csv'.format(sys.argv[2]))
    data=data.readlines()
    data = np.array([data[i].strip().split(',') for i in range(1,len(data))])
    test_img_file = data[:, 0]
    test_label = data[:, 1].astype(np.float32)
    
    train = np.array([ np.loadtxt('{}/{}'.format(sys.argv[1],k)) for k in train_img_file])
    
    test = np.array([ np.loadtxt('{}/{}'.format(sys.argv[2],k)) for k in test_img_file])
    
    convolutional_filter=(2,2)
    
    model = firstCNN(train, train_label, test, test_label, convolutional_filter, stride = 1,eta = 0.01)
    
    model.model_train()
    
    model.accuracy_test() 

import numpy as np
import sys


#read Data
train = open(sys.argv[1])
data = np.loadtxt(train)
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0], 1))
train = np.append(train, onearray, axis=1)

test = open(sys.argv[2])
data = np.loadtxt(test)
test = data[:,1:]
testlabels = data[:,0]

print(test)
rows=train.shape[0]
cols=train.shape[1]

hidden_nodes=int(sys.argv[3])

##intialize weights

w=np.random.rand(hidden_nodes)
W=np.random.rand(hidden_nodes,cols)
hidden_layer=np.matmul(train,np.transpose(W))

#activation 
sigmoid=lambda x: 1/(1+np.exp(-x))

#multiplication of sigmoid and hidden layer
hidden_layer=np.array([sigmoid(i) for i in hidden_layer])

#multiplication of hidden layer and output layer
output_layer=np.matmul(hidden_layer,np.transpose(w))

#Calculate objective
obj=np.sum(np.square(output_layer-trainlabels))

index=np.arange(train.shape[0])

#print(hidden_layer)
epochs=10000
eta=.01
stop=0
i=0
m=int(sys.argv[4])
prev_obj=np.inf
while(i<epochs):
   
    prev_obj=obj
    np.random.shuffle(index)
    first_k_lst=index[:m]
    
   #updating weights(w)
    dellw=np.zeros((hidden_nodes))
    for j in first_k_lst:
        dellw+=(np.dot(hidden_layer[j],np.transpose(w))-trainlabels[j])*hidden_layer[j]
    w=w-eta*dellw
#    print('w',w)
    #updating weights for hidden layer
    dellW=np.zeros((hidden_nodes, cols))
    for k in range(hidden_nodes):
        for j in first_k_lst:
         dellW[k]+=(np.dot(hidden_layer[j,:],w)-trainlabels[j]) * w[k]  * (hidden_layer[j,k] ) *(1-hidden_layer[j,k]) *train[j]
    
#    print('W:',dellW)
    W=W-eta*dellW
    
#    print('hidden nodes',W)
    #recalculate the objective
    
    hidden_layer=np.matmul(train,np.transpose(W))

    hidden_layer=np.array([sigmoid(i) for i in hidden_layer])
    
    #print(hidden_layer) 
   
    output_layer=np.matmul(hidden_layer,np.transpose(w))
    obj=np.sum(np.square(output_layer-trainlabels))
    print('obj',obj)
    i=i+1

onearray=np.ones((test.shape[0],1))
test=np.append(test,onearray,axis=1)

result=np.matmul(test,np.transpose(W))
result=np.array([sigmoid(i) for i in result])
result=np.dot(result,np.transpose(w))

result=np.sign(result)


for i, prd in enumerate(result):
    print(i, prd)

print('\n')
count = 0
for prd, tl in zip(result, testlabels):
    if prd !=tl:
        count += 1

print("Number of wrong predictions :", count)

print("\nAccuracy...")
print((1 - sum(abs(0.5*(testlabels - result)))/testlabels.shape[0]) * 100)
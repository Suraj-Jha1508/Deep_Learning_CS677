# Deep_Learning_CS677

This Repository contain all Assignments and Project i completed during my Deep Learning Course at NJIT

# Assignment 1: CUDA program for computing the dot product

Solution : https://github.com/Suraj-Jha1508/Deep_Learning_CS677/tree/master/Assignment_1

Write a CUDA program for computing the dot product of a vector in parallel with 
each row of a matrix. You are required to have each thread access consecutive
memory locations (coalescent memory access). The inputs are 

1. number of rows
2. number of columns
3. a data matrix file similar to the format in the Chi2 program 
4. a vector file (one row)
5. cuda device
6. number of threads

For example if the input is

1 2 0
1 1 0
1 2 1

and w = (2, 4, 6)

then your program should output

10
6
16

Compute the dot products in parallel your kernel function. You will have to
transpose the data matrix in order to get coalescent memory access. 


# Assignment 2: CUDA program into an OpenMP

Convert the CUDA program that you wrote for assignment one into an
OpenMP one. The output of both your CUDA and OpenML programs must be the same. 

# Assignment 3: Python program that trains a single layer neural network with sigmoid activation

Write a Python program that trains a single layer neural network
with sigmoid activation. You may use numpy. Your input is in dense 
liblinear format which means you exclude the dimension and include 0's. 

Let your program command line be:

python single_layer_nn.py <train> <test> <n>

where n is the number of nodes in the single hidden layer.

For this assignment you basically have to implement gradient
descent. Use the update equations we derived on our google document
shared with the class.

Test your program on the XOR dataset:

1 0 0
1 1 1
-1 0 1
-1 1 0

1. Does your network reach 0 training error? 

2. Can you make your program into stochastic gradient descent (SGD)?

3. Does SGD give lower test error than full gradient descent?

4. What happens if change the activation to sign? Will the same algorithm
work? If not what will you change to make the algorithm converge to a local
minimum?

# Assignment 4: Implement stochastic gradient descent

Implement stochastic gradient descent in your back propagation program
that you wrote in assignment 3. We will do the mini-batch SGD search. 

I. Mini-batch SGD algorithm:

Initialize random weights
for(k = 0 to n_epochs):
	Shuffle the rows (or row indices)
	for j = 0 to rows-1:
		Select the first k datapoints where k is the mini-batch size
		Determine gradient using just the selected k datapoints
		Update weights with gradient
	Recalculate objective

Your input, output, and command line parameters are the same as assignment 3.
We take the batch size k as input. We leave the offset for the final layer 
to be zero at this time.

Test your program on the XOR dataset:

1 0 0
1 1 1
-1 0 1
-1 1 0

1. Test your program on breast cancer and ionosphere given on the website. Is the 
mini-batch faster or the original one? How about accuracy?

2. Is the search faster or more accurate if you keep track of the best objective
in the inner loop?

# Assignment 5: Python program that trains a neural network

Write a Python program that trains a neural network with a single 2x2
convolutional layer with stride 1 and global average pooling. See
our course notes on google drive for equation updates with sigmoid
activation. 

The input are 3x3 images. Images for training are going to be in
one directory called train and test ones in the directory called
test. The train directory has a csv file called data.csv that contains
the name of each image dataset and its label. For example your data.csv
would look like

Name,Label
image0.txt,1
image1.txt,-1

where image0.txt is 

1 0 0
0 1 0
0 0 1

and image1.txt is 

0 0 1
0 1 0
1 0 0

# Assignment 6: convolutional network in Keras to train the Mini-ImageNet 

Write a convolutional network in Keras to train the Mini-ImageNet 
dataset on the course website. Your constraint is to create a network
that achieves at least 80% test accuracy (in order to get full points).

Submit your assignments as two files train.py and test.py. Make
train.py take three inputs: the input training data, training labels,
and a model file name to save the model to. 

python train.py <train.npy> <trainlablels.npy> <model file>

It is straightforward to save a Keras model to file, see the simple example here 
https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state

Make test.py take three inputs: the input test data, test labels,
and a model file name to load the model. 

python test.py <test.npy> <testlabels.npy> <model file>

The output of test.py is the test error of the data which is
the number of misclassifications divided by size of the test set.

# Assignment 7: convolutional network in Keras to train the Mini-ImageNet

Write a convolutional network in Keras to train the Mini-ImageNet 
dataset on the course website. You may use transfer learning. Your
goal is to achieve above 90% accuracy on the test/validation datasets.

Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the input training directory
and a model file name to save the model to.

python train.py train <model file>

Make test.py take two inputs: the test directory
and a model file name to load the model.

python test.py test <model file>

The output of test.py is the test error of the data which is
the number of misclassifications divided by size of the test set.

# Assignment 8: Classify images in the three Kaggle datasets on the course website with convolutional networks

Classify images in the three Kaggle datasets on the course website 
with convolutional networks. You may use transfer learning. Your
goal is to achieve above 85% accuracy on the test/validation datasets.

Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the input training directory
and a model file name to save the model to. 

python train.py train <model file>

Make test.py take three inputs: the test directory 
and a model file name to load the model. 

python test.py test <model file>

The output of test.py is the test error of the data which is
the number of misclassifications divided by size of the test set.

# Assignment 9: Implement a simple GAN in Keras

Implement a simple GAN in Keras to generate MNIST images. Use the GAN given here

https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

as your discriminator and generator. 

You want to train the generator to produce images of numbers between 0 and 9.

Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the input training directory
and a model file name to save the generator model to. 

python train.py MNIST_train_directory <generator model file>

Make test.py take one input: the generator model file. The output
of test.py should be images resembling MNIST digits saved to the output
file.

# Assignment 10: Implement a simple black box attack in Keras

Implement a simple black box attack in Keras to attack a pretrained 
ResNet18 model from Keras. For the substitute model we use a two hidden 
layer neural network with each layer having 100 nodes.

Our goal is to generate adversaries to decieve a simple single layer 
neural network with 20 hidden nodes into misclassifying data from a 
test set that is provided by us. This test set consists of examples 
from classes 0 and 1 from CIFAR10. 

Your target model should have at least 85% accuracy on the test set without
adversaries. 

A successful attack should have a classification accuracy of at most 10%
on the test.

Submit your assignments as two files train.py and test.py. Make
train.py take three inputs: the test data, the target model to 
attack (in our case this is the network with 20 hidden nodes),
and a model file name to save the black box model file to.

python train.py <test set> <target model to be attacked> <black box model file> 

Your train.py program should output the accuracy of the target model on the
test data without adversaries as the first step. This is to verify that your
model has high accuracy on the test data without adversaries. Otherwise if your
model has low test accuracy it will be harder to attack.

When running train.py output the accuracy of the target model on the adversaries 
generated from the test data after each epoch.

Make test.py take three inputs: test set, target model, and the black box model.
The output should be the accuracy of adversarial examples generated with
epsilon=0.0625. A successful submission will have accuracy below 10%
on the advsersarial examples.

# Assignment 11: word2vec model from fake news dataset

Learn a word2vec model from fake news dataset and a real news dataset. We 
will use the word2vec model implemented in the Python Gensim library. Now 
we have two sets of word representations learnt from different datasets. 

Output the top 5 most similar words to the following ones from each 
representation.

1. Hillary
2. Trump
3. Obama
4. Immigration

In order to do this we first normalize all vector representations (set them 
to Euclidean length 1). Consider the vector x for a given word w. We 
compare the cosine similarity between x and the vectors x' for each word w' 
in the fake news dataset first. We then output the top 5 words with highest 
similarity. We then do the same for the real news and then see if the top 
similar words differ considerably.

Submit your assignments as two files train.py and test.py. Make
train.py take two inputs: the text dataset on which to learn the words and
a model file name to save the word2vec model to.

python train.py <text data> <word2vec model file to save the model> 

Make test.py take three inputs: text dataset, word2vec model, a query file 
containing five query words. The output should be the top five most similar 
words to each word in the query file.


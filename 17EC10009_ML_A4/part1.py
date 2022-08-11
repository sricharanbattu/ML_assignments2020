# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:10:25 2020

@author: lenovo
B.Sricharan
17EC10009,sricharanbattu@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



"""_______________________________THE FOLLOWING FUNCTIONS ARE USED FOR PROCESSING AND LOADING THE DATA____________________________"""
def preprocessor(filename='./data/datasetnew.txt'):
    """
    This function converts the text file into two csv files:
    a train set and a test set after one hot encoding the output values.
    The train(80% of all data) and test(20% of all data) instances are sampled randomly.
    """
    data=pd.read_csv(filename,delimiter='\t',header=None)
    cols=data.columns
    n=len(cols)
    data['out1']=(data[cols[-1]]==1)+0
    data['out2']=(data[cols[-1]]==2)+0
    data['out3']=(data[cols[-1]]==3)+0
    for i in range(0,n-1):
        mu=data[cols[i]].mean()
        std=data[cols[i]].std()
        data[cols[i]]=(data[cols[i]]-mu)/std
    data.to_csv('./data/datasetnew.csv',index=False)
    data = data.sample(frac=1).reset_index(drop=True)
    train=data.sample(frac=0.8,random_state=200)            #randomisation of the data and picking 80% of data 
    test=data.drop(train.index)
    train.to_csv('./data/train.csv',index=False)
    test.to_csv('./data/test.csv',index=False)
    
def dataloader(traindata,minibatchsize=32):
    """
    This function takes the data(a csv file) and 
    converts it into required number of batches and their corresponding outs.
    The output is a tuple of list:The first list is the batches of features and 
    the second is the batches list of outputs
    """
    train=np.genfromtxt(traindata,delimiter=',')
    train=train[1:,]
    rows=train.shape[0]
    minibatches=[]
    outs=[]
    firstindex=0;
    lastindex=0
    count=1;
    while(lastindex<rows):
        lastindex=count*minibatchsize;
        trainmini=train[firstindex:min(lastindex,rows),:-4]
        outmini=train[firstindex:min(lastindex,rows),-3:]
        minibatches.append(trainmini)
        outs.append(outmini)
        firstindex=lastindex
        count=count+1
    return minibatches,outs 


"""______________________________________THE FOLLOWING DEFINES A NEURAL NET___________________________________________________"""

def forward(nn,inputs):
    """
    A forward pass is done in the neural network i.e outputs of the neurons
    are updated based on the previous weights and the present inputs
    """
    nn.update_outputs(inputs)
    
def backpropagation(nn,true_outs,eta):
    """
    The weights of the neural network are updated using
    backpropagation algorithm,which is implemented in the
    member functions of the neural network
    """
    nn.update_deltas(true_outs)
    nn.update_weights(eta)

    
def weight_initialise(mini,maxi,rows,cols):
    """
    initialises the weights for the neurons in the neural network,
    with random weights
    """
    return np.random.uniform(low=mini,high=maxi,size=(rows,cols))

def prediction_outs(preds):
    """
    The output class is predicted based on the outer layer values of 
    the neural network
    """
    x=np.argmax(preds,axis=1)+1
    return x

def accuracy(preds,actuals):
    """
    This function returns the proportion of the correct values predicted
    """
    return np.mean(preds==actuals)


    
class layers:
    """
    This class is an entire neural network,specifically denoting a single layer
    The class has member attributes:
            1.weightmatrix: each column denotes the weights corresponding to a single neuron
                for the inputs from previous layer
            2.deltas: each row contains the derivative of the output wrt the net.
                each row corresponds to an instance in the batch
            3.width:the width of the layer or no of neurons in this layer
            4.inputs: the inputs for this layer/outputs of previous layer.This is for calculation ease while training
            5.outputs: the outputs of this layer.This is for calculation ease
            6.actfunc: The activation function of a neuron. All neurons in this layer has same activation function
            7.layertype: whether the layer is hidden layer or output layer
            8.child: The next layer if this layer is a hidden layer
    This class has member functions:
            1.update_outputs: The attribute outputs is updated during every iteration
            2.update_deltas : The attribute delta is modified during every iteration
            3.update_weights: The attribute weightmatrix is modified during every iteration
                The above two functions implement the back propagation algorithm
            4.predict       : The trained network gives predictions for a set of data
    """
    def __init__(self,layerwidth,layerid,activationhidden,activationouter='softmax'):
        """
       
       The instantiation function of the network takes arguments:
           1.layerwidth: A list denoting the nof of neurons in successive layers including the input and output layer (However,we don't construct input layer)
           2.layerid : which position is the layer present in.
           3.activation hidden : the activation function for the neurons in the hidden layers(sigmoid/ReLU). All hidden layer neurons have same activation function
           4.activationouter: The activation function for output layer. The default is softmax.
        """
        self.weightmatrix=weight_initialise(-1,1,layerwidth[layerid-1],layerwidth[layerid])
        self.deltas=np.zeros(layerwidth[layerid])
        self.width=layerwidth[layerid]
        self.inputs=np.zeros(layerwidth[layerid-1])
        self.outputs=np.zeros(layerwidth[layerid])
        if(layerid==len(layerwidth)-1):
            self.actfunc=activationouter
            self.layertype='output'
            
        else:
            self.actfunc=activationhidden
            self.layertype='hidden'
            self.child=layers(layerwidth,layerid+1,activationhidden,activationouter)
            
            
    def update_outputs(self,inputs):
        """
        Based on the weights of the neurons, when an input is passed,the outputs are 
        successively updated for neurons in each layer, starting from the first layer
        The outputs are activated based on the activation function.
        """
        self.inputs=inputs
        x=np.dot(inputs,self.weightmatrix)
        if(self.actfunc=='sigmoid'):
            self.outputs=1/(1+np.exp(-1*x))
        elif(self.actfunc=='relu'):
            x[x<0]=0
            self.outputs=x
        elif(self.actfunc=='softmax'):
            x=np.exp(x)
            summed=x.sum(axis=1)
            summed=np.diag(1/summed)
            self.outputs=np.dot(summed,x)
            
        if(self.layertype=='hidden'):
            self.child.update_outputs(self.outputs)
            
            
    def update_deltas(self,true_outs):
        """
        BackPropagation Algorithm is implemented in this function. The derivatives
        wrt to the nets are stored in the delta attribute. Each training instance has a delta 
        set associated with it.The derivatives are based on the activation function. For the outer layer
        categorical cross entropy Loss function is employed.The back propagation is a backward process i.e 
        the updation starts from the output layer and then propagates to first layer.
        """
        if(self.layertype=='output'):
            
            self.delta=-1*(self.outputs-true_outs)
            return
        else:
            if(self.actfunc=='sigmoid'):
                self.child.update_deltas(true_outs)
                weights=np.transpose(self.child.weightmatrix)
                self.delta=self.outputs*(1-self.outputs)*np.dot(self.child.delta,weights)
            elif(self.actfunc=='relu'):
                self.child.update_deltas(true_outs)
                weights=np.transpose(self.child.weightmatrix)
                self.delta=(self.outputs>0)*np.dot(self.child.delta,weights)
                
                
    def update_weights(self,eta):
        """
        The weights of neurons in each layer are updated based on the updated deltas.
        It could be implemented ina forward manner or a backward manner. We did it in forward way.
        """
        N=self.inputs.shape[0]
        inps=np.transpose(self.inputs)
        delweights=(1/N)*eta*np.dot(inps,self.delta)
        self.weightmatrix=self.weightmatrix+delweights
        if(self.layertype=='hidden'):
            self.child.update_weights(eta)
            
    def predict(self,data):
        """
        The trained network is used to predict the outputs when some inputs are given.The outputs are calculated
        by multiplying with the weights and passing through the activation function of the neuron. The outputs of this hidden layer acts as
        inputs to the next layer. Thus the final outputs are the outputs of the output layer.These are probabilities corresponding to
        different classes,rather than the actual classes.
        """
        if(self.layertype=='output'):
            data=np.dot(data,self.weightmatrix)
            data=np.exp(data)
            data[np.isnan(data)]=100
            summed=data.sum(axis=1)
            summed=np.diag(1/summed)
            x=np.dot(summed,data)
            return x
        else:
            data=np.dot(data,self.weightmatrix)
            if(self.actfunc=='sigmoid'):
                data=1/(1+np.exp(-1*data))
            elif(self.actfunc=='relu'):
                data[data<0]=0
            return self.child.predict(data)


"""_________________________________________THE FOLLOWING FUNCTIONS ARE USED FOR IMPLEMENTING THE MACHINE LEARNING STEPS________________________________"""
def train_neuralnet(nn,trainbatches,trainouts,eta):
    """
    10 epochs are implemented during the training of the neural network in this function.
    The inputs are the features and outputs of the training data segregated as batches, and learning rate
    """
    no_batches=len(trainbatches)
    for i in range(0,10):
        for j in range(0,no_batches):
            forward(nn,trainbatches[j])
            backpropagation(nn,trainouts[j],eta)
                

        
def predict_neuralnet(nn,feature):
    """
    The class predictions are made based on the probabilities given by the neural network
    when the input data are passed to it
    """
    pred_probs=nn.predict(feature)
    preds=prediction_outs(pred_probs)
    return preds

def TrainAndPredict(nn,trainbatches,trainouts,trainfeature,trainout,testfeature,testout):
    """
    This functio implements all machinelearning steps i.e Training of the network with traindata,predicting the train labels and test labels
    However prediction is done after every 10 epochs and the accuracies of train and test data are returned for plotting
    """
    superepoch=0
    trainsetaccuracy=[]
    testsetaccuracy=[]
    while(superepoch<20):
        superepoch=superepoch+1
        train_neuralnet(nn,trainbatches,trainouts,eta)
        train_preds=predict_neuralnet(nn,trainfeature)
        test_preds=predict_neuralnet(nn,testfeature)
        trainsetaccuracy.append(accuracy(train_preds,trainout))
        testsetaccuracy.append(accuracy(test_preds,testout))
    return trainsetaccuracy,testsetaccuracy


def plot_accuracies(trainsetaccuracy,testsetaccuracy,qno):
    """
    Train accuracy and test accuracy are plotted against the number of epochs ,which is 
    equivalent(but different) to observing the accuracy as a function of number of iterations
    """
    x=np.arange(10,210,10)
    fig=plt.figure(figsize=(10,10))
    ax=fig.gca()
    plt.plot(x,trainsetaccuracy,color='red',label='TrainAccuracy')
    plt.plot(x,testsetaccuracy,color='blue',label='TestAccuracy')
    ax.set_xticks(np.arange(0,210,10))
    plt.title(qno+" Plot of accuracies with a neural net against an epoch ")
    plt.xlabel("Number of epochs")
    plt.ylabel("accuracy")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()
    
    
"""_____________________________________________ACTUAL IMPLEMENTATION IS DONE BELOW USING THE ABOVE UTILITIES__________________________________"""   
eta=0.01
preprocessor('./data/datasetnew.txt')                                  #The text data is converted to train and test
minibatches=32
trainsets=dataloader('./data/train.csv',minibatches)                   # The train data is divided into batches and is output as features and outputs
trainbatches=trainsets[0]
trainouts=trainsets[1]

trainset=np.genfromtxt('./data/train.csv',delimiter=',')
trainfeature=trainset[1:,:-4]
trainout=trainset[1:,7]

testset=np.genfromtxt('./data/test.csv',delimiter=',')
testfeature=testset[1:,:-4]
testout=testset[1:,7]



featuresize=testfeature.shape[1]
layerwidth1=[featuresize,32,3]                                      #no of neurons in each layer as required by neural net in part1a
layerwidth2=[featuresize,64,32,3]                                   # as required by neural net in part1b
nn1=layers(layerwidth1,1,'sigmoid','softmax')                       # a neural net is instantiated as required by part1a
nn2=layers(layerwidth2,1,'relu','softmax')                          # another neural net is instantiated as required by part1b

acc1=TrainAndPredict(nn1,trainbatches,trainouts,trainfeature,trainout,testfeature,testout)
trainacc1=np.array(acc1[0])
testacc1=np.array(acc1[1])
acc2=TrainAndPredict(nn2,trainbatches,trainouts,trainfeature,trainout,testfeature,testout)
trainacc2=np.array(acc2[0])
testacc2=np.array(acc2[1])

plot_accuracies(trainacc1,testacc1,'Part 1A')
print("_"*120)
print("                                     Part 1A:")
print("_"*120)
#x=np.argmax(testacc1)
print("Final Train Accuray            :                        {0:0.4f}".format(trainacc1[-1]))
print("Final Test Accuracy            :                        {0:0.4f}".format(testacc1[-1]))

print("_"*120)
plot_accuracies(trainacc2,testacc2,'Part 1b')

print("                                   Part 1B:")
print("_"*120)
#x=np.argmax(testacc2)
print("Final Train Accuracy           :                       {0:.4f}".format(trainacc2[-1]))
print("Final Test Accuracy            :                       {0:0.4f}".format(testacc2[-1]))
print("_"*120)






  
       
    
    


# Hello! It is recommended to read the README file to understand how this code works actually..

import time #Nothing to do here, Just to calculate the execution time, we will use later
import numpy as np



start = time.time() # To calculate the execution time


# Here, I developed the Neural Network to have 3 layers, L1,L2, output -- I do not consider the inputs as a Layer!
# You can modify the numbers of the Neurons of each Layer, just change the variables below..

neuronsL1 = 9   #Neurons of the First layer, I set it to 9 
neuronsL2 = 5   #Neurons of the Second layer, I set it to 5
neuronsL3 = 1   #Neurons of the Last layer (Output Layer) , I set it to 1. Because it is a Binary Classification Problem


#Define the Activation Function [I used Sigmoid Here.] -- You can use any other function you want
def sigmoid(x):
    return 1.0 / ( 1+np.exp(-x) )


#Define the derivative of the used Activation Function 
def sigmoid_derivative(x):
    return x * ( 1.0 - x )



# Our Neural Network, Most of our work will be here..
class NeuralNetwork:
    def __init__(self,inp,oup):
# I created variables for the input data [ Features ] and labels [ The True Y or the real Y output ] -- I am preparing the variables now



# inp = Input [Features]  -- we have 8 rows represent the dataset [How many examples do we have] & 10 columns which are our Features

        self.inp = inp  # inp = Input [Examples , Features] The matrix dimansions are   8 * 10
        self.oup = oup  # oup = Output [ True Y or Orginal labled output]         7 * 1
        
        
#       Create random values for our weights                          
        self.weights1 = np.random.rand(neuronsL1 , self.inp.shape[1] )           # 9 * 10   [How many neurons you have in the first  layer] * [ How many features ]      
        self.weights2 = np.random.rand(neuronsL2 , neuronsL1 )                   # 5 * 9    [How many neurons you have in the second layer] * [ How many rows in the previeous layer ] 
        self.weights3 = np.random.rand(neuronsL3 , neuronsL2 )                   # 1 * 5    [How many neurons you have in the third (output) layer] * [ How many rows in the previeous layer ]
        
        
        
#       Initializing the Biases Values...
#       I created biases for all possible multiplication of W and X  -- I am not sure if this right. Or, it must be a bias for every Neuron, remain Constant and does not change with every training Example like here
#       EX: ( W[0,0] dot X[000] ) + B[0,0] + ( W[0,0] dot X[0,1,0] ) + B[0,1] 

        self.b1       =  np.random.rand( neuronsL1 , self.inp.shape[0]  )        #  9 * 8
        self.b2       =  np.random.rand( neuronsL2 , self.inp.shape[0]  )        #  5 * 8
        self.b3       =  np.random.rand( neuronsL3 , self.inp.shape[0]  )        #  1 * 8


    def feedforward(self): 
#       The equation that I use is : a(z) = a(     { W[L] dot a[L-1] }  + Bias   )  --- a = output of the layer | W = weights 

        self.layer1   = sigmoid(   np.dot(    self.weights1 , self.inp.T  ) + self.b1    )   # 9 * 8
        self.layer2   = sigmoid(   np.dot(    self.weights2 , self.layer1 ) + self.b2    )   # 5 * 8
        self.expected = sigmoid(   np.dot(    self.weights3 , self.layer2 ) + self.b3    ).T # 8 * 1  --- I get the Transpose of it to fit the next step of the Dot Product


        
    def backward(self):         
        
#       The Equation that I use for Backpropagation is written in the README file, check it if you did not understand what is coming...
#       Sometimes I get the transpose of a matrix (which is not supposed to be, at least according to the equation I use, but, I made this to be able to multiplicate matrices with proper dims. ) I do not know of this right. However, it seems to be working in with these data
        
#                              8 * 1          8 * 1                             8 * 1
        d_finall   =  2 * ( self.expected - self.oup ) * sigmoid_derivative(self.expected)   #  8 * 1   ---  d_finall = delta of the finall layer [Output]
        
        
#                                         5 * 8                 5 * 1           1 * 8                                                      
        d_weights2 = sigmoid_derivative(self.layer2) * np.dot(self.weights3.T, d_finall.T)    #  5 * 8          
        
        
#                                         9 * 8                   9 * 5            5 *  8                      
        d_weights1 = sigmoid_derivative(self.layer1)  * np.dot( self.weights2.T , d_weights2 ) #  9 * 8
        
        
        
#       updating Biases.... 
        self.b3    -=  d_finall.T 
        self.b2    -=  d_weights2
        self.b1    -=  d_weights1
#       If you do not know why Biases equal this, Check the README file      

        
        
#       updating Weights....
#                                      1 * 8        8 * 5                                     
        self.weights3 -=    np.dot(d_finall.T , self.layer2.T ) 
        
#                                      5 * 8        8 * 9                             
        self.weights2 -=    np.dot( d_weights2 , self.layer1.T )
        
#                                      9 * 8      8 * 10                             
        self.weights1 -=    np.dot( d_weights1 , self.inp )
#       Any transpose here do not follow the equation I just created it to be able to multiply it with dot product method
        
        
        
# Imagine it as subjects in school Algebra, Calculus, Physics etc... and we have 8 students with different grades for 10 subjects
inp = np.array( [ [1,1,0,1,1,1,1,0,1,1], 
                  [0,1,1,1,1,0,1,0,1,0],
                  [1,0,1,0,1,1,0,0,1,1], 
                  
                  [1,0,0,1,0,0,0,1,0,1],
                  [0,0,1,1,1,0,1,0,0,0],
                  [1,0,0,1,0,1,0,0,1,1],
                  
                  [1,1,1,0,1,0,0,1,1,0],
                  [0,1,0,0,1,0,0,1,0,0] ] )   #  8 * 10


oup = np.array([ [1], 
                 [1], 
                 [1], 
                 [0],
                 [0],
                 [0],
                 [1],
                 [1] ])                       #  8 * 1


nn = NeuralNetwork(inp, oup)

for i in range(10000):
# It is essential to follow the Feedforward path first, then the backward
    nn.feedforward()
    nn.backward()

# To calculate the Execution time
end = time.time()



# Just print the weights to see it, but you can ignore it. It is just to understand the data more

print('\n---\nWeights 1')
print(nn.weights1)

print('\n---\nWeights 2')
print(nn.weights2)

print('\n---\nWeights 3')
print(nn.weights3)

print('\n---\n')

# Our predicted values
print('The Expected Values are :')
print(nn.expected)
print('====')

print('Error Per Element')
print( oup - nn.expected)
print('\n====')
 
# The equation is =  ( [ Y(Original or the True values) - Y(hat or predicted) ]^2 ) / How many Samples do we have |
print('The Total Error Is')
print( .5    *    ( np.sum(    (oup - nn.expected)  ** 2   )    /   oup.shape[0]     )       )
print('\n====')

#The Execution time. You can use any other method you want
extim = end-start

print('Excuation time is =>', extim , 'Seconds<=')






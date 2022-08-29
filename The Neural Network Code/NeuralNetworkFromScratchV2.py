# Hello! It is recommended to read the README file to understand how this code works actually...

import time #Nothing to do here, just to calculate the execution time, we will use later
import numpy as np
        
       
# OUr very Simple data set.
# Imagine it as subjects in school Algebra, Calculus, Physics etc... and we have 6 students with different grades for 3 subjects
inp =  np.array([ [1,1,1],[0,0,0],[1,0,1],
                  [1,0,0],[1,1,0],[0,0,1] ])

oup = np.array( [  [1],[0],[1],
                   [0],[1],[0]  ] )

inptst = np.array(  [  [0,1,1]  ] ) # Just to train it

start = time.time() # To calculate the execution time


# Here, I developed the Neural Network to have 3 layers, L1, L2, output -- I do not consider the inputs as a Layer!
# You can modify the numbers of the Neurons of each Layer, just change the variables below.

neuronsL1 = 2   #Neurons of the First layer, I set it to 2
neuronsL2 = 2   #Neurons of the Second layer, I set it to 2
neuronsL3 = 1   #Neurons of the Last layer (Output Layer), I set it to 1. Because it is a Binary Classification Problem

  

#----------------------------------------------------------------------------------------

# The Activation Function of the last layer (the output layer)
def sigmoid(x):
    return 1.0 / ( 1+np.exp(-x) )


#Define the derivative of the used Activation Function 
def sigmoid_derivative(x):
    return x * ( 1.0 - x )

#Define the Activation Function [I used LRelu Here.] -- You can use any other function you want
def relu(x):
    return  0.01* np.maximum(x,0)


#Define the derivative of the used Activation Function 
def relu_derivative(x):
        x[x<=0] *= 0.01 
        x[x>0 ] = 1
        return x

#----------------------------------------------------------------------------------------


# Our Neural Network, Most of our work will be here.
class NeuralNetwork:
    def __init__(self,inp,oup, inptst):
        
# I created variables for the input data [ Features] and labels [ The True Y or the real Y output ] -- I am preparing the variables now
# inp = Input [Features]  -- we have 6 rows represent the dataset [How many examples do we have] & 3 columns which are our Features

        self.inp = inp  # inp = Input [Examples , Features] The matrix dimensions are     6 3 
        self.oup = oup  # oup = Output [ True Y or Original labeled output]               6 1
        
#       Our training samples
        self.inptst = inptst 
        
#       Create random values for our weights                          
        self.weights1 = np.random.rand(neuronsL1 , self.inp.shape[1] )           # 2 3    [How many neurons you have in the first  layer] * [ How many features ]      
        self.weights2 = np.random.rand(neuronsL2 , neuronsL1 )                   # 2 2    [How many neurons you have in the second layer] * [ How many rows in the previous layer ] 
        self.weights3 = np.random.rand(neuronsL3 , neuronsL2 )                   # 1 2    [How many neurons you have in the third (output) layer] * [ How many rows in the previous layer ]
        
        
        
#       Initializing the Biases Values...
        self.b1       =  np.random.rand( neuronsL1 , 1 )        #  2  1
        self.b2       =  np.random.rand( neuronsL2 , 1 )        #  2  1
        self.b3       =  np.random.rand( neuronsL3 , 1 )        #  1  1


#--------------------------

    def feedforward(self): 
#       The equation that I use is: a(z) = a(     { W[L] dot a[L-1] }  + Bias   )  --- a = output of the layer | W = weights 
#                                                  2 3            3 6                   
        self.layer1   = relu(      np.dot(    self.weights1 , self.inp.T  ) + self.b1    )   # 2 6
#                                                  2 2            2  6                   
        self.layer2   = relu(      np.dot(    self.weights2 , self.layer1 ) + self.b2    )   # 2 6
#                                                 1 2             2 6                   
        self.expected = sigmoid(   np.dot(    self.weights3 , self.layer2 ) + self.b3    )   # 1 6  

#--------------------------
        
        
    def backward(self):         
        
#       The Equation that I use for Backpropagation is written in the README file, check it if you did not understand what is coming...
#       Sometimes I get the transpose of a matrix (which is not supposed to be, at least according to the equation I use, but, I made this to be able to multiplicate matrices with proper dims. ) I do not know of this right. However, it seems to be working in with these data
        
#                              1  6        1  6                               1  6                              
        d_finall   =  ( (self.expected - self.oup.T) / ( self.expected * ( 1 - self.expected )   )  * sigmoid_derivative(self.expected )) #  1   6   ---  d_finall = delta of the finall layer [Output]
#                                                        2  2          2  6         2  1                   2 1          1  6                                                      
        d_weights2 = relu_derivative(  np.dot(    self.weights2 , self.layer1 ) + self.b2 ) * np.dot(self.weights3.T, d_finall)           #  2   6

#                                                        2  3          3  6          2  1                         2  2            2  6
        d_weights1 = relu_derivative(   np.dot(    self.weights1 , self.inp.T  ) + self.b1    )  * np.dot( self.weights2.T , d_weights2 ) #  2   6
        
#       Alpha Value
        r = 0.1        

#       If you do not know why Biases equal this, Check the README file      

#------

#       updating Weights....
#                                    1  6        6  2                                    
        self.weights3 -=  ( np.dot(d_finall , self.layer2.T   ) * r ) 
        
#                                   2 6              6  2                            
        self.weights2 -= ( np.dot( d_weights2 , self.layer1.T ) * r ) 
        
#                                     2  6          6  3                             
        self.weights1 -=  ( np.dot( d_weights1 , self.inp     ) * r ) 
#       Any transpose here might not follow the equation I just created it to be able to multiply it with dot product method
        
#       Updating Biases.... 
#       I summed all the rows of the different Examples. So, I can get one bias per every neuron no matter how many training instances you have, and to be able to apply it to any test examples
      
        self.b3     -=  np.sum(  d_finall,axis=1 ,  keepdims= True )  * r
#                        2 1        2  10
        self.b2     -=  np.sum(  d_weights2, axis=1 , keepdims= True)  * r
#                        3 1         3  10
        self.b1     -=  np.sum( d_weights1, axis= 1,  keepdims=True ) * r
      

#--------------------------


        
    def feedforwardTST(self): 

#       Just to test the Neural Network
        self.layer1   = relu(         np.dot(    self.weights1 , self.inptst.T  ) + self.b1    )
#                                                  4 5            5 10                   
        self.layer2   = relu(         np.dot(    self.weights2 , self.layer1 )    + self.b2    )  

        self.expectedTST = sigmoid(   np.dot(    self.weights3 , self.layer2 )    + self.b3    )




#----------------------------------------------------------------------------------------



nn = NeuralNetwork(inp, oup,inptst)

#Test whether our weights and biases are updated or not, this helped me to know if the algorithm work
print('Bias 3 Before\n===-=-=-=-=',nn.b3)
print('Weights3 Before\n===-=-=-=-=',nn.weights3)


for i in range(1010):
# It is essential to follow the Feedforward path first, then the backward
    nn.feedforward()
    nn.backward()



print('Bias 3 After\n===-=-=-=-=' ,nn.b3)
print('Weights3 After\n===-=-=-=-=',nn.weights3)
print('\nThe Expected Train Values:')
print(nn.expected.T)
print('====\n')


print('\nThe Expected TESTED Values:')
nn.feedforwardTST()
print(nn.expectedTST.T)
print('====\n')


#------


# I used Cross Entropy Loss Function 
print('The Total Error train Is')
err = (-1/nn.oup.shape[0]) * np.sum( ( oup* np.log(nn.expected)  )   + (   (1-oup) * np.log(1-nn.expected)  )  )
print(   err   )


#------


# To calculate the Execution time
end = time.time()
extim = end-start
print('Excuation time is =>', extim , 'Seconds<=')









<h1 align="center">Simple Neural Network From Scratch</h1>
<div>
  <p align="center">
    In this project, I break down the basic idea of Neural Networks without any libraries -other than NumPy- just simple to understand how the NN works in the background. I used NumPy to handle the Linear Algebra part, dealing with matrices, multiplying them, and so on. 
    <br/>
  </p>
</div>

<br/>
<div align="center">
  <a href="https://i.imgur.com/uSRVsmx.png">
    <img src="https://i.imgur.com/uSRVsmx.png" alt="Logo" width="620" height="320">
  </a>

<br/>
</div>

## Tech Stack

* **Python:** Version 3.10

* **NumPy:** Version 1.23.0

* **Spyder IDE**: Version 5.3.2

## Details

* After I watched a Youtube course about Neural Networks and when I came to the part to apply what I have learned, I got stuck with that! I was not able to code a simple NN by myself, I was just copying the code with no understanding of how it works. I searched here and there until I got the point. The idea is not difficult so I recommend you to see the References below for a better understanding.
<br/>

* You can Imagine this problem as a Student Affairs Specialist, we have 6 students with their grades in 3 subjects (e.g. Math, Physics, Music, Arabic Langauge... Chiemsrty). if the student passed specific subjects, will he pass to next year or he will get failed and repeat the year? We assume that 1 = passed the subjects and 0 = Did not pass the subject. Also, in the results, 1 = passed the year successfully, and 0 = failed the year (Did not pass it). So, our goal is to determine if a student with X grades in Y subjects will pass the year. This is our problem.
<br/>

* The subjects are called Features which are the columns, and the rows which are how many students we have it is 8 in this example, and of course, you can add or delete some data. The backpropagation equation is written in the video in the References check it.
<br/>

* My Neural Network is supposed to solve a Binary Classification Problem, I created the dataset as an example to focus more on the core idea. However, you can add your dataset and it should work fine! I made the NN with 3 Layers(Not including the Input Layer), the first layer has 9 Neurons and the second layer has 5 Neurons and The last Layer(Output Layer) has 1 Neuron. In addition, I used the __Sigmoid function__ as an Activation Function. (Please See figure 1 below to understand the Diagram of the NN).
<br/>


* Why did I choose these specific numbers of layers and neurons? I do not know! Till now, I still do not know how to set these Parameters in a reasonable way not just choosing them without any sense! I keep searching about that and once I understand itو I shall update this file with the clarification.
<br/>


## Update Version V2.0

* __Well... I have found many bugs and incorrect concepts I have applied here. I will list them here so you can avoid them.__
<br/>
<br/>

* The first mistake, I did not know that for a Binary Classification Problem it's recommended to use a __Leaky Relu (Or Relu)__ as an Activation Function in the hidden layers while in the Output layer I have to use the __Sigmoid Function__. So, simply I just applied these concepts but doing this __Only__ did not solve the problem.
<br/>

* Secondly, I applied the derivatives of the Activation Function improperly :( Thus, the expected outputs did not make any sense. When I give it a look again, solved it by hand. So, it became more reasonable.
<br/>

* Thirdly, I did not multiply the weights and biases by the __Factor alpha α or eta η (Learning rate)__ So it leads to __NaN__ Values or Constant values(i.e. The same output for all instances)  
<br/>

* Fourth, I used biases for every neuron for every sample- which is not correct indeed. the right thing is to make a bias for every neuron. Not for every example per every neuron. I used __np.sum()__ to sum the whole row per training examples so finally I will have a vector of biases n*1 where n is number is the Neurons number
<br/>

* Finally, I used a very simple dataset I just created as I have also changed the neurons number (but the figures still working tho) just to test the algorithm. However, when I tried __Haberman's Survival Data Set__ it did not work. I do not know why, so, I think this might need some __PreProcessing Data__ before we could start training. I do not know how to do that yet, but I will keep this updated if I managed to do it. __You will find the dataset filtered and sorted randomly in the Reposoitry__ 
<br/>


<br/>

## Figures
* [Figure 1 - Neural Network Diagram - High Quality](https://i.imgur.com/ZYPjIko.png) 
 <div align="center">
  <a href="https://i.imgur.com/ZYPjIko.png">
    <img src="https://i.imgur.com/ZYPjIko.png" alt="Figure 1">
  </a>
</div>
<br/>
<br/>

* [Figure 2 - Equations - High Quality](https://i.imgur.com/ZYPjIko.png) 
 <div align="center">
  <a href="https://i.imgur.com/qgOu7dk.png">
    <img src="https://i.imgur.com/qgOu7dk.png" alt="Figure 2">
  </a>
  </div>
<br/>


## Contributing
Contributions are what makes the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Do not forget to give the project a star! Thanks again!

<br/>

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.





## References

*  This is important [lecture](https://www.youtube.com/watch?v=3UFgAaJibjg), as it illustrates how to deal with Neural Networks using Linear Algebra, and finally, how to code it. __Watching is a MUST!__

*  I recommend this book because it shows you the concept behind Neural Networks with very interactive and intuitive animations. Plus, it explains how to build Neural Networks on your own, without any libraries, so you can better understand deep learning and how all of the elements work - Very good to understand the basics of Neural Networks. __[Neural Networks from Scratch in Python
by Harrison Kinsley, Daniel Kukieła](https://nnfs.io/)__

*  The good thing about these 2 articles specifically, is the MATH and Partial Derivative explained behind them. If you do not understand the Math (__Which is really important indeed__) behind it, then you must read these 2 articles. [A Derivation of Backpropagation in Matrix Form](https://sudeepraja.github.io/Neural/) -  And - [Neural Networks: Feedforward and Backpropagation Explained & Optimization](https://mlfromscratch.com/neural-networks-explained/#/)


## Contacts
* Via Email : Mahmoud.Nady@Ejust.edu.eg
* [Via FaceBook]( https://www.facebook.com/MND919/ ).








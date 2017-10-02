Assignment 1
===============

**Simple Linear Regression**

You are given the following training data.

| X | Y |
| --- | --- |
| 2.0 | 5.1 |
| 2.5 | 6.1 |
| 3.0 | 6.9 |
| 3.5 | 7.8 |
| 4.0 | 9.2 |
| 4.5 | 9.9 |
| 5.0 | 11.5 |
| 5.5 | 12.0 |
| 6.0 | 12.8 |

1. Model Y as a linear function of X.
2. Use gradient descent learning algorithm to learn model parameters for α = 0.01, 0.1, and 1.0.  Use an appropriate convergence criterion. Plot J for the learning duration for each case.

**Multiple Linear Regression**

You are given the following training data.

 | X1 | X2  | Y |
| --- | --- | --- |
| 2.0 | 70.0 | 79.4 |
| 3.0 | 30.0 | 41.5 |
| 4.0 | 80.0 | 97.5 |
| 4.0 | 20.0 | 36.1 |
| 3.0 | 50.0 | 63.2 |
| 7.0 | 10.0 | 39.5 |
| 5.0 | 50.0 | 69.8 |
| 3.0 | 90.0 | 103.5 |
| 2.0 | 20.0 | 29.5 |

1. Model Y as a linear function of X1 and X2.  Pick an appropriate value for the learning rate.
2. Learn the parameters of the regression surface without scaling the predictor variables.
3. Learn the parameters of the regression surface after scaling the predictor variables (divide by range).


Assignment 2
===============

A group of 20 students studied 0 to 6 hours for the exam.  Some passed and others failed. Results are given below (Data is taken from Wikipedia).

| Student | Hours studied  - x | Result (0 – fail, 1 – pass) - y |
| --- | --- | --- |
| 1 | 0.5 | 0 |
| 2 | 0.75 | 0 |
| 3 | 1.00 | 0 |
| 4 | 1.25 | 0 |
| 5 | 1.50 | 0 |
| 6 | 1.75 | 0 |
| 7 | 1.75 | 1 |
| 8 | 2.00 | 0 |
| 9 | 2.25 | 1 |
| 10 | 2.50 | 0 |
| 11 | 2.75 | 1 |
| 12 | 3.00 | 0 |
| 13 | 3.25 | 1 |
| 14 | 3.50 | 0 |
| 15 | 4.00 | 1 |
| 16 | 4.25 | 1 |
| 17 | 4.50 | 1 |
| 18 | 4.75 | 1 |
| 19 | 5.00 | 1 |
| 20 | 5.50 | 1 |

1. Determine the optimal linear hypothesis using linear regression to predict if a student passes or not based on the number hours studied.
2. Determine the optimal logistic hypothesis using logistic regression to predict if a student passes or not based on the number hours studied.
3. Plot both hypothesis function 0 &lt; x &lt; 6. Compare and explain the two results obtained.
4. Develop a logistic regression-like algorithm for the following cost function.
    Y = 1 - Cost function goes from 100 to 0 linearly as hypothesis function goes from 0 to 1
    Y = 0 - Cost function goes from 0 to 100 linearly as hypothesis function goes from 0 to 1
Compare results with those of the standard logistic algorithm.


Assignment 3
============

1. You are given 21 well distributed training pattern of dimensionality 3.
    1. What is the probability that a linear classifier separates a randomly selected dichotomy of training patterns?
    2.  What is the minimum degree polynomial function that guarantees separation of any dichotomy of 21 sample patterns?
2. You are given training patterns {X1 , .  .  .  .  , XN} from both classes. All patterns are augmented by 1 and patterns 
of Class-2 are multiplied by -1 as discussed in the class.  The cost function
    J(θ) = ∑ # I = 1 to N (1/X # i # t X # i) [| θ # t X # i | # 2 | θ # t X # i | θ # t X # i ]
attains its only minimum value of zero if θ # t X # i &gt; 0 for all X # i.  Develop a machine learning algorithm based on gradient
descent approach to learn θ from training data to separate the two classes.

3. You are given the following training data.
    Class1: (2  4) # t , (3  3) # t
    Class2: (6 12) # t , (8  10) # t

Starting with an initial parameter vector [0 1 1] # t ,
    1. Illustrate 4 iterations of perceptron – formulation 1.
    2. Illustrate 4 iterations of perceptron – formulation 2.
    3. Illustrate 4 iterations of relaxation with b = 1.
    4. Illustrate 4 iterations of Ho-Kashyap algorithm with b = [1 1 1 1] # t.


Assignment 4
============

In this assignment you will design and train a backpropagation network to learn the
exclusive-OR function.
1. Use one hidden layer with 4 hidden units. Plot the error as a function of the number of iterations. Remember the initial weights.
2. Verify that momentum term indeed improves convergence (μ = 0.5). Use the initial weights remembered in (1).
3. Repeat (1) using bipolar sigmoid and bipolar representation for exclusive-OR function. The network should converge faster.
4. Verify that Nguyen-Widrow approach of assigning initial weights improves convergence.
5. Use two hidden layers (3 units in the first hidden layer and 2 units in the second hidden layer).

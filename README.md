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

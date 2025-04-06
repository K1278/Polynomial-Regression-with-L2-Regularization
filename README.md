# Polynomial-Regression-with-L2-Regularization
Polynomial regression with L2 regularization minimizing bias-variance trade-off by optimizing hyperparameters

The dataset is contained in a file labeled "Data.csv".
It is a synthetic dataset comsisting of 1000 datapoints, each having one feature variable and one continuous target variable.

Both, the _ipynb_ file and the _py_ file contain the same code. 

Following is what the code does:
1. Loads the dataset into a Pandas dataframe
2. Normalises the feature variable by the formula X' = (X - mean) / SD
3. Splits and shuffles the data into three sets - training, cross-validatation, and testing.
4. Visualizes the split data using scatter plots.
5. Trains 9 simple polynomial regression models (with degrees 1-9) using batch gradient descent for 500 iterations.
6. Visualises the simple regression polynomials.
7. Adds L2 regularization and builds 9 new models.
8. Experiments with different values of lambda.
9. Calculates and plots the bias (square), variance, and total error for each lambda.
10. Finds the optimal lambda using Bias-Variance decomposition.
11. Plots the best polynomial fit on the datapoints.

Before running the code, ensure that the filepath of the dataset is updated in the Python code. 

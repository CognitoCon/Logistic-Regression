# Logistic-Regression-Classifier

  This is a class-based implementation of a regularized logistic regression classifier in Python using Numpy and pandas. In its implementation it takes a pandas dataframe where every column except for the last is assumed to be a feature and the final column is assumed to be a classification (1 or 0).
  
  Included is "main.py" which provides an example of how logistic regression can be performed using the logReg class. The program first takes a training file and a test file as inputs. Then, it creates and trains a model using the training data given along with optional parameters including learning rate, number of iterations, and a regularization term. It then provides a projected 2D plot of the data (color-coded by classification) along with the boundary produced by the training algorithm.

![Training Data](/train.png)

  Although the algorithm works with any number of features including combinations and powers thereof, the plot2dProjection class function only acts meaningfully when the model is trained on features of degree 1. Given this present restriction the sample data included only includes two features even though the data was created using a boundary based on the product of both features.

  Finally, the program generates a simulated model using the boundary parameters produced by the training data along with the actual results of the test data. During this generation a class containing statistical information about the fit is produced. This data includes accuracy, precision, recall, F1, and F2 scores. This data is printed and then the 2D projection of the test data along with the aforementioned boundary is plotted. 
  
  ![Training Data](/test.png)

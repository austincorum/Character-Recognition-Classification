# CS599_P1
Character recognition task: classification of handwritten numerals


Problem Setting:
The inputs that make up this figure are inputs for each Network based on the number of input pixels (256 = 16x16 for Net-1). There are 10 outputs in this model for each of the digits classes from an image (class k, for k = 0,1,2,....,9). The desired function to learn is the predicted value fˆk(x) .

Data Sources:
The U.S. Postal service uses images that are scanned, resulting in 16x16 greyscale images that need to be classified. With each network, there are misclassification rates around 4.5%. This is the real problem that this figure solves. This graph represents the correctness of test data, over the number of training iterations. There were five different networks that were examined. The first network overfits very quick, where as the second network overfits slightly over the course of the iterations. Net-5 shows the best classification results, based on the highest percent of correct data. I plan to get this data from the Elements of Statistical Learning website (https://web.stanford.edu/~hastie/ElemStatLearn/). There are 7291 training observations(X inputs) and 2007 test observations(Y outputs). Each of these data sets are split into rows/lines. 7291 rows in the training data and 2007 rows in the testing data.

Algorithm(Baselines):
I’m going to have a for loop through every network, every row in the training iterations, and every row in the data for the correctness(%) of the test data. Each network will be plotted, one by one. The predicted values for each network will be calculated, then plotted. If I chose this figure, I will learn how to create a neural network for character recognition. I plan on using PyTorch or TensorFlow for this project, as this is what I am most familiar with, the Python language. I should be able to find packages for these within the python regression package that will allow me to utilize the predicted value function.

# MachineLearningExercises
This repository contains some machine learning algorithms that I have worked on during the semester.

#### Bernoulli Naive Bayes
This folder contains the implementation of a Bernoulli Naive Bayes model with no smoothing on data found in the training dataset. Practical1.1.py contains the Naive Bayes model, and its output for the training dataset provided is stored in class_priors.tsv, negative_feature_likelihoods.tsv and positive_feature_likelihoods.tsv The autograder_test_NB.py file is there to check the accuracy of the model using the validation set in the folder.

#### Logistic Regression
This folder implements a Logistic Regression model using gradient descent on data found in the training dataset. Practical1.2.py contains the logistic regression model, and outputs the file weights.tsv when the training dataset provided is used. The autograder_test_LR.py file is there to check the accuracy of the model using the validation set in the folder.

#### K-Means
Implementation of K-means model on data from the Iris dataset with 3 clusters (dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Iris), also included as Data.tsv in the folder). The K-means algorithm is implemented in the k-means.py file and the file containing the clustering for this dataset is saved as kmeans_output.tsv. autograderClustering.py is the program used to test the accuracy of the model.

#### GMM
Implementation of a Gaussian Mixture Model on data from the Iris dataset with 3 clusters (dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Iris), also included as Data.tsv in the folder). The GMM algorithm is implemented in the gmm.py file and the file containing the clustering for this dataset is saved as gmm_output.tsv. autograderClustering.py is used to test the accuracy of the model.

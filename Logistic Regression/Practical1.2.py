#importing NUMPY
import numpy
import math
import csv
#implementing logistic regression
class LogReg():
    def __init__(self):
        #dictionary of attributes to store parameters of the model
        self.weights = 0

    def fit(self, X, y, max_iters = 1000, lr = 0.00025, to1 = 0.0001):
        self.weights = numpy.zeros((len(X[0])+1))
        self.weights[0] =1
        updatedweights = numpy.zeros(len(self.weights))
        i = 0
        while (numpy.linalg.norm(numpy.subtract(updatedweights, self.weights)) >= to1) and (max_iters >= i):
            updatedweights[0] = 1
            if i!=0:
                self.weights = updatedweights
            i = i+1
            for k in range(len(self.weights)-1):
                sumTerm =0
                for j in range(len(y)):
                    sigmak = sigmoid(self.weights[k] * X[j][k])
                    multTerm = y[j] - sigmak
                    sumTerm = sumTerm + (X[j][k] *multTerm)
                updatedweights[k+1] = self.weights[k+1]+ (lr * sumTerm)

    def save_parameters(self):
        saveTSV("weights.tsv", self.weights)

#helper method to find sigma of a certain term. Takes in matrix and returns each element sigmoided
def sigmoid(l):
    return 1 / (1 + math.exp(-l))

#saving as tsv file (for class priors, positive feature likelihoods and negative feature likelihoods)
def saveTSV(name, filetxt):
    numpy.savetxt( name, filetxt, delimiter = '\t')
    print("Saved!")

#importing data
data = numpy.loadtxt("train_dataset.tsv", delimiter="\t")
#separating features and targets
X = data[:,:-1] #which selects all the rows and all the columns except the last one.
y = data[:,-1] #which selects all the rows and only the last column.
#calling methods for logistic regression
logisticreg = LogReg()
logisticreg.fit(X, y)
logisticreg.save_parameters()
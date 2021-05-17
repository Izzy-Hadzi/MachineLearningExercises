#importing NUMPY
import numpy
import csv
#implementing Naive Bayes class
class Naive():
    def __init__(self):
        #dictionary of attributes to store parameters of the model
        self.cps = []
        self.positive_features = []
        self.negative_features = []

    #which takes the training data as input and sets the model parameter attributes as a side eﬀect. 
    def fit(self, X, y):
        #finding feature likelihoods
        numofy1 =0
        numperfeaturepos = numpy.zeros(len(X[0]), float)
        numperfeatureneg= numpy.zeros(len(X[0]), float)
        for i in range(len(y)):
            #class prior
            if y[i] ==1:
                numofy1= numofy1 +1
                #features of that vector
                for j in range(len(X[i])):
                    numperfeaturepos[j] = numperfeaturepos[j] + X[i][j]
            else:
                for j in range(len(X[i])):
                    numperfeatureneg[j] = numperfeatureneg[j] + X[i][j]
        #setting class priors
        self.cps.append(numofy1/len(y))
        self.cps.append(1-(numofy1/len(y)))
        self.cps = numpy.array(self.cps)
        #setting feature likelihood
        for i in range(len(numperfeaturepos)):
            self.positive_features.append(numperfeaturepos[i]/numofy1)
            self.negative_features.append(numperfeatureneg[i]/(len(y)-numofy1))
        self.positive_features = numpy.array(self.positive_features)
        self.negative_features = numpy.array(self.negative_features)
    
    #Should save files in same directory
    def save_parameters(self):
        saveTSV("class_priors.tsv", self.cps) 
        saveTSV("negative_feature_likelihoods.tsv", self.negative_features)
        saveTSV("positive_feature_likelihoods.tsv", self.positive_features) 

#saving as tsv file (for class priors, positive feature likelihoods and negative feature likelihoods)
def saveTSV(name, filetxt):
    numpy.savetxt( name, filetxt, delimiter = '\t')

#importing data
data = numpy.loadtxt("train_dataset.tsv", delimiter="\t")
#separating features and targets
X = data[:,:-1] #which selects all the rows and all the columns except the last one.
y = data[:,-1] #which selects all the rows and only the last column.
#calling methods for Naive Bayes
naivebayes = Naive()
naivebayes.fit(X,y)
naivebayes.save_parameters()

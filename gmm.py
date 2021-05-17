import numpy as np
from scipy.stats import multivariate_normal

def gmm(X, prev):
    #inializing subparts if kmenas-output is not provided
    #first expectation step, separating into 3 equal subparts
    #subparts = np.array_split(X,3)
    #initializing arrays from means assignments
    subparts = [[] for y in range(3)]
    for i in range(len(prev)):
        if(prev[i] == 0.0):
            subparts[0].append(X[i])
        elif (prev[i] == 1.0):
            subparts[1].append(X[i])
        else:
            subparts[2].append(X[i])
    #means
    means = [mean(subparts[0]), mean(subparts[1]), mean(subparts[2])]
    #covariances
    covariances = [cov(subparts[0], means[0]), cov(subparts[1],means[1]), cov(subparts[2],means[2])]
    #setting weights vector
    pi_vector = [0.3333, 0.3333, 0.3333]
    threshold = 0.00001
    #defining new log likelihood and log likelihood
    logl = loglikelihood(X, pi_vector, means, covariances)
    newlogl = 10000
    # run until difference in log likelihoods <10^-5
    while (abs(newlogl - logl) > threshold):
        #update logl at each iteration except the first
        if newlogl != 10000:
            logl = newlogl
        #expectation step
        scores = np.zeros((len(X), 3))
        for i in range(len(X)):
            scores[i] = rscore(X[i], pi_vector, means, covariances)
        #maximization step
        means= updatemeans(scores, X)
        covariances = updatecovariances(scores, X, means)
        pi_vector = updatepivalues(scores)
        newlogl = loglikelihood(X, pi_vector, means, covariances)
    #setting labels to highest probability
    labels = np.zeros(len(X))
    for i in range(len(scores)):
        labels[i] = np.argmax(scores[i])
    return labels

#HELPER METHODS
def mean(data):
    m= np.zeros(4, dtype = float)
    for i in range(len(data)):
        m[0] = m[0] + data[i][0]
        m[1] = m[1] + data[i][1]
        m[2] = m[2]+ data[i][2]
        m[3] = m[3]+ data[i][3]
    m[0] = m[0]/len(data)
    m[1] = m[1]/len(data)
    m[2] = m[2]/len(data)
    m[3] = m[3]/len(data)
    return m

def cov(data, mean):
    #making a 4x4 covariance matrix for each cluster
    bottom = len(data)-1
    c= np.zeros((4,4))
    for i in data:
        term = np.matrix(i-mean)
        term2 = np.transpose(np.matrix(term))
        step = np.dot(term2, term)
        c = c + step
    c=c/bottom
    return c

def loglikelihood(data, pi_vector, means, covariances):
    logl = 0.0
    for i in range(len(data)):
        term=0.0
        for j in range(len(means)):
            normaldistribution = multivariate_normal.pdf(data[i], means[j], covariances[j])
            term = term+ (normaldistribution * pi_vector[j])
        logl = logl +np.math.log(term)
    return logl

def rscore(point, pi_values, means, covariances):
    #score matrix having k=0, k=1, k=2 
    score = np.zeros(3) 
    top0 = pi_values[0] * multivariate_normal.pdf(point, means[0], covariances[0])
    top1 = pi_values[1] * multivariate_normal.pdf(point, means[1], covariances[1])
    top2 = pi_values[2] * multivariate_normal.pdf(point, means[2], covariances[2])
    bottom = top0 + top1 +top2
    score[0] = top0/bottom
    score[1] = top1/bottom
    score[2] = top2/bottom
    return score

def updatemeans(scores, data):
    t0 =0.0
    t1=0.0
    t2=0.0
    m0 = np.zeros(4)
    m1 = np.zeros(4)
    m2 = np.zeros(4)
    for i in range(len(scores)):
        t0 = t0+scores[i][0]
        t1 = t1+scores[i][1]
        t2 = t2+scores[i][2]
        m0 = m0 + scores[i][0]*data[i]
        m1 = m1+ scores[i][1]*data[i]
        m2 = m2 +scores[i][2]*data[i]
    m0 = m0/t0
    m1 = m1/t1
    m2 = m2/t2
    means = [m0, m1, m2]
    return means

def updatecovariances(scores, data, means):
    t0 =0.0
    t1=0.0
    t2=0.0
    cov0= np.zeros((4,4))
    cov1 = np.zeros((4,4))
    cov2 =np.zeros((4,4))
    for i in range(len(scores)):
        t0 = t0+scores[i][0]
        t1 = t1+scores[i][1]
        t2 = t2+scores[i][2]
        cov0 = cov0 + scores[i][0] * np.dot( np.transpose(np.matrix(data[i]-means[0])), np.matrix(data[i]-means[0]))
        cov1 = cov1 + scores[i][1] * np.dot( np.transpose(np.matrix(data[i]-means[1])), np.matrix(data[i]-means[1]))
        cov2 =  cov2 + scores[i][2] * np.dot( np.transpose(np.matrix(data[i]-means[2])), np.matrix(data[i]-means[2]))
    cov0 = cov0/t0
    cov1 = cov1/t1
    cov2 = cov2/t2
    covariances = [cov0, cov1, cov2]
    return covariances

def updatepivalues(scores,):
    total = len(scores)
    pi_values = np.zeros(3)
    for i in scores:
        pi_values[0] = pi_values[0] + i[0]
        pi_values[1] = pi_values[1] + i[1]
        pi_values[2] = pi_values[2] + i[2]
    pi_values[0] = pi_values[0]/total
    pi_values[1] = pi_values[1]/total
    pi_values[2] = pi_values[2]/total
    return pi_values


#MAIN METHOD
#take in data from tsv file
X = np.genfromtxt('Data.tsv', delimiter='\t')
prev = np.genfromtxt('kmeans_output.tsv', delimiter='\t')
#running gmm algorithm 
final_labels = gmm(X, prev)

#output file as tsv gmm_output.tsv
output_array = np.asarray(final_labels)
np.savetxt("gmm_output.tsv", output_array, delimiter = '\t')
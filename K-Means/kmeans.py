import numpy as np

#running the kmeans algorithm described in section 14.6 of the notes
def kmeans(centroids, X):
    #new centroids is the updates version of the centroids
    new_centroids = [[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]]
    #loop and update centroids until there is no change in the centroids
    while (new_centroids != centroids):
        if (new_centroids != [[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]] ):
            centroids = new_centroids
        #assigning all points to cluster
        labels = np.zeros(len(X))
        for i in range(len(X)):
            label = findclosest(centroids, X[i])
            labels[i] = label
        #recomputing centroids
        new_centroids = computecentroid(labels, X)
    print(new_centroids)
    return labels

#HELPER METHODS
#finding cluster that a point is closest to
def findclosest(centroids, point):
    p = np.asarray(point)
    c0 = np.asarray(centroids[0])
    c1 = np.asarray(centroids[1])
    c2 = np.asarray(centroids[2])
    #assigned is the cluster its assigned to, it can be 0, 1 or 2
    assigned = 0
    #compute distances 
    d0 =np.linalg.norm(c0-p)
    d1 =np.linalg.norm(c1-p)
    d2 =np.linalg.norm(c2-p)
    #assigned to certain cluster
    if(d0<d1 and d0<d2):
        assigned = 0
    elif (d1<d0 and d1<d2):
        assigned =1
    else:
        assigned =2
    return assigned

def computecentroid(labels, data):
    #compute the new centroids
    t0=0.0
    c0=[0.0,0.0,0.0,0.0]
    t1 =0.0
    c1=[0.0,0.0,0.0,0.0]
    t2=0.0
    c2=[0.0,0.0,0.0,0.0]
    #finding total number of points in cluster and adding values of each cluster to that row
    for i in range(len(labels)):
        if(labels[i]==0):
            c0[0] = c0[0] + data[i][0]
            c0[1] = c0[1] + data[i][1]
            c0[2] = c0[2] + data[i][2]
            c0[3] = c0[3] + data[i][3]
            t0 = t0+1
        elif(labels[i]==1):
            c1[0] = c1[0] + data[i][0]
            c1[1] = c1[1] + data[i][1]
            c1[2] = c1[2] + data[i][2]
            c1[3] = c1[3] + data[i][3]
            t1 = t1+1
        else:
            c2[0] = c2[0] + data[i][0]
            c2[1] = c2[1] + data[i][1]
            c2[2] = c2[2] + data[i][2]
            c2[3] = c2[3] + data[i][3]
            t2 = t2+1
    for i in range(4):
        if t0 == 0:
            c0[i] = 0.0
        else:
            c0[i] = c0[i]/t0
        if t1==0:
            c1[i] =0.0
        else:
            c1[i] = c1[i]/t1
        if t2 ==0:
            c2[i] = 0.0
        else:
            c2[i] = c2[i]/t2
    computed_centroids = [c0, c1, c2]
    return computed_centroids

# MAIN METHOD
#take in data from tsv file
X = np.genfromtxt('Data.tsv', delimiter='\t')

# initialize centroids
centroids = [[ 1.03800476, 0.09821729, 1.0469454, 1.58046376],[ 0.18982966, -1.97355361, 0.70592084, 0.3957741 ], [ 1.2803405, 0.09821729, 0.76275827, 1.44883158]]

#running k-means algorithm on data set
final_labels = kmeans(centroids, X)
#output file as tsv kmeans_output.tsv
output_array = np.asarray(final_labels)
np.savetxt("kmeans_output.tsv", output_array, delimiter = '\t')
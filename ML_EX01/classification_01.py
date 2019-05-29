import matplotlib.pyplot as plt #import pyplot in order to plot
from sklearn.datasets import load_iris #load dataset from sklearn specificaly the iris dataset
from sklearn.preprocessing import Normalizer

import numpy as np

data = load_iris() #assign the data form the iris dataset
features = data['data'] #Take the data labeled as 'data' and apply it to features
feature_names = data['feature_names']
target = data['target']

for t,marker,c in zip(range(3),">ox","rgb"):
    plt.scatter(features[target == t,0],
                features[target == t,1],
                marker=marker,
                c=c)
    plt.show()

#building first classification model
plength = features[:,2]

#getting setosa flower features
target_names = data['target_names']
labels = target_names[target]
is_setosa = (labels == "setosa")
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()

print("Maximum of setosa : {0}.".format(max_setosa))
print("Minimum of others : {0}.".format(min_non_setosa))
print("------------------------")

for x in features[:,2]:
    if x < 2:
        print("Iris Setosa")
    else:
        print("Iris  Virginica or Iris Versicolour")


#step 2 now select non-setosa features
features = features[~is_setosa]
labels = labels[~is_setosa]#non - setosa labeled
virginica = (labels == 'virginica') #from those non - setosa labeled vector pick the ones labeled virginica

best_acc = -1.0
for fi in range(features.shape[1]):
    print("features.shape[1] = {0} fi value {1}".format(features.shape[1],fi))

    #Generate all possible threshold for this feature
    thresh = features[:,fi].copy() #generate a shallow copy of


    thresh.sort()
    for n in thresh:
        print(n)

    # Now test all thresholds:
    for t in thresh:
        pred = (features[:,fi] > t)
        acc = (pred == virginica).mean()
        if acc > best_acc:#if accuracy is better than best accuracy make new best current accuracy
            best_acc = acc
            best_fi = fi
            best_t = t





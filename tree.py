from sklearn.cluster import KMeans
import pandas as p
import numpy as np

# read the csv
data = p.read_csv('ww.csv')

# change DataFrame to list
data_list = []
for i in data.iterrows():
    data_list.append(list(i[1][:]))

# break data_list to data_label and data_fearure
data_list = np.array(data_list, dtype=np.int)
data_label = data_list[..., 0]
data_feature = data_list[..., 1:]
del data_list, data

class MyTree:
    def __init__(self):
        self.k2 = KMeans(n_clusters=2)
        self.weights = []
    def fit(self, X, y):
        'set the features in X and set the labels in y'
        self.features = X
        self.labels = y
        self.find_center()
        return self
    def find_center(self):
        'data with the same label is in on cluster and exchage with the center cluster'
        labels = len(set(self.labels))
        self.new_label = [i for i in range(1,labels+1)]
        labels = [[] for i in range(labels)]
        for i,j in enumerate(self.labels):
            labels[j-1].append(list(self.features[i]))
        self.labels = []
        k = KMeans(n_clusters=1)
        for i in labels:
            k.fit(i)
            self.labels.append(list(k.cluster_centers_.flat))
    def tree(self, list_raw):
        if len(list_raw) != 1:
            list= (np.array(list_raw)*np.array(self.wiegh(list_raw))).tolist()
            self.k2.fit(list)
            self.weights.append(self.wiegh(list_raw))
            first = []
            second = []
            for i, j in enumerate(list_raw):
                if self.k2.labels_[i] == 0:
                    first.append(j)
                else:
                    second.append(j)
            first = self.tree(first)
            second = self.tree(second)
            return [first, second]
        return np.array(np.array(list_raw).flat).tolist()
    def wiegh(self, list):
        list = np.array(list)
        weigh = []
        for i in range(len(list[0])):
            avg = sum(list[..., i])/len(list)
            diff = []
            for j in list[..., i]:
                diff.append(abs(avg-j))
            weigh.append(sum(diff)/len(diff))
        return weigh
a = MyTree().fit(data_feature, data_label)
print(a.tree(a.labels))
print(a.weights)
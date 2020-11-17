import numpy as np
from tqdm import trange
from sklearn.neighbors._kd_tree import KDTree


class MLKNN(object):
    '''
    Usage:
        model = MLKNN(n_labels=n_labels, n_neighbours=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    '''

    def __init__(self, n_labels, n_neighbours=10, smooth=1.0):
        '''
            Parameters:
                n_labels - Number of labels
                n_neighbours - Number of considered neighbours
                smooth - Laplace smooth constant
        '''
        self.n_labels = n_labels
        self.n_neighbours = n_neighbours
        self.smooth = smooth
        self.X, self.y = None, None
        self.Ph, self.Peh = None, None
        self.KDTree = None

    def knn(self, target, K, exclude=[]):
        dist, index = self.KDTree.query(target.reshape(1, -1), k=K + len(exclude))
        ret, index = [], index.ravel()
        for i in index:
            if len(ret) >= K: break
            if i in exclude: continue
            ret.append(i)
        return np.array(ret)

    def fit(self, X, y):
        self.X, self.y = X, y
        self.Ph = np.zeros((self.n_labels, 2))
        self.Peh = np.zeros((self.n_labels, self.n_neighbours + 2, 2))

        # Calculate the prior probabilities
        label_range = trange(self.n_labels)
        label_range.set_description("Calculate the prior probabilities")
        for l in label_range:
            self.Ph[l, 1] = (self.smooth + np.sum(self.y[:, l])) / (self.X.shape[0] + self.smooth + self.smooth)
            self.Ph[l, 0] = 1.0 - self.Ph[l, 1]

        # Calculate the posterior probabilities
        label_range = trange(self.n_labels)
        label_range.set_description("Calculate the posterior probabilities")
        self.KDTree = KDTree(self.X, leaf_size=2, metric="euclidean")
        for l in label_range:
            c = np.zeros((self.n_neighbours + 1, 2))
            for i in range(self.X.shape[0]):
                neighbour = self.knn(self.X[i, :], self.n_neighbours, exclude=[i])
                delta = np.sum(self.y[neighbour, l])
                c[delta, self.y[i, l]] += 1
            for j in range(self.n_neighbours + 1):
                self.Peh[l, j, 1] = (self.smooth + c[j, 1]) / (self.smooth * (self.n_neighbours + 1) + np.sum(c[:, 1]))
                self.Peh[l, j, 0] = (self.smooth + c[j, 0]) / (self.smooth * (self.n_neighbours + 1) + np.sum(c[:, 0]))
        return self

    def predict_proba(self, X):
        if self.KDTree is None: raise Exception("Have not been trained.")
        proba = []
        for x in X:
            neighbour = self.knn(x, self.n_neighbours)
            label = []
            for l in range(self.n_labels):
                delta = np.sum(self.y[neighbour, l])
                P0, P1 = self.Ph[l, 0] * self.Peh[l, delta, 0], self.Ph[l, 1] * self.Peh[l, delta, 1]
                if P1 + P0 == 0.0:
                    label.append(self.Ph[l, 1])
                else:
                    label.append(P1 / (P1 + P0))
            proba.append(label)
        return np.array(proba)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, 0)

    '''Notice: Load and Save functions have not been tested yet.'''
    def save(self, path):
        np.savez(
            path,
            X=self.X,
            n_neighbour=self.n_neighbours,
            n_labels=self.n_labels,
            Ph=self.Ph,
            Peh=self.Peh,
        )

    def load(self, path):
        raw = np.load(path)
        self.X = raw["X"]
        self.KDTree = KDTree(self.X, leaf_size=2, metric="euclidean")
        self.n_neighbours = raw["n_neighbours"]
        self.n_labels = raw["n_labels"]
        self.Ph = raw["Ph"]
        self.Peh = raw["Peh"]
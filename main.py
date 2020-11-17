import arff
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from methods.MLKNN import MLKNN
from metrics import hamming_loss

def main(data="yeast", algo="MLKNN"):
    if data == "yeast": n_features, n_labels, path = 103, 14, "./dataset/yeast_corpus/yeast.arff"

    # Read
    data = arff.load(open(path))
    data = np.array(data["data"]).astype(float)
    X, y = data[:, :n_features], data[:, -n_labels:].astype(int)

    # Cross Validation
    hamming = []
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    for trI, teI in kfold.split(X, y):
        # Train & Test
        X_train, y_train = X[trI, :], y[trI, :]
        X_test,  y_test  = X[teI, :], y[teI, :]
        if algo == "MLKNN": model = MLKNN(n_labels=n_labels, n_neighbours=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Evaluate
        hamming.append(hamming_loss(y_test, y_pred))
        print(hamming[-1])
    print("%f, %f" % (np.mean(hamming), np.std(hamming)))

if __name__ == '__main__': main()

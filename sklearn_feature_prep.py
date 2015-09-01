import pickle
import pandas as pd
import numpy as np


def load_features():
    with open('intermediate/sklearn_features.pkl', 'rb') as f:
        docids, features = pickle.load(f)

    # temporary
    from os.path import basename
    docids = np.asarray([basename(d) for d in docids[1:]])
    features = features.tocsr()[1:,:]
    # end temporary

    all_targets = pd.read_csv('data/train_no_holdout.csv', index_col='file')

    id_in_train = np.in1d(docids, all_targets.index)
    train_features = features[id_in_train, :]

    train_targets = all_targets.loc[docids[id_in_train]]

    return np.asarray(train_targets.index), train_features, train_targets['sponsored']

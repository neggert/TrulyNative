import click
import pandas as pd
import numpy as np
from numpy.random import laplace
from sklearn.metrics import roc_auc_score

def load_predictions(predfile):
    print(predfile)
    p = pd.read_csv(predfile, header=False, delim_whitespace=True,
                    names=('raw_pred', 'filename'), index_col=1)
    p['pred'] = 1. / (1 + np.exp(-1. * p['raw_pred']))
    return p


def load_truth(truthfile):
    return pd.read_csv(truthfile, index_col=0,
                       names=('filename', 'target'))

def get_score(pred_df, truth_df):
    df = truth_df.join(pred_df, how='left')
    return roc_auc_score(df['target'], df['pred'])


def thresholdout(train, holdout, threshold, tolerance):
    if np.abs(train - holdout) < threshold + laplace(scale=tolerance):
        return train
    else:
        return holdout + laplace(scale=tolerance)


@click.command()
@click.argument('predfile', type=click.Path(exists=True, dir_okay=False))
def main(predfile):
    preds = load_predictions(predfile)
    train_truth = load_truth('data/train_no_holdout.csv')
    holdout_truth = load_truth('data/holdout.csv')

    train_score = get_score(preds, train_truth)
    holdout_score = get_score(preds, holdout_truth)

    threshold = 0.001
    tolerance = threshold / 4

    score = thresholdout(train_score, holdout_score, threshold, tolerance)

    print(score)

    with open('log.txt', 'a') as f:
        f.write(', '.join((predfile, str(score))))


if __name__ == '__main__':
    main()

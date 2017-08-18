#!/usr/bin/env python

"""
This python script takes the data produced by the extract_features script
and trains some scikit-learn classifiers to distinguish signal from background.
"""

# System imports
import argparse
import logging
import pickle

# External imports
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('train_classifiers')
    add_arg = parser.add_argument
    add_arg('input_file', help='Input npz file of training features')
    add_arg('output_file', help='Output pkl file for the trained classifiers')
    return parser.parse_args()

def main():
    """Main execution function"""
    # Command line parsing
    args = parse_args()
    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Initializing')
    logging.info('Configuration: %s' % args)

    # Load the input data
    with np.load(args.input_file) as f:
        X = f['X']
        y = f['y']
        passSR = f['passSR']
        weights = f['weights']

    logging.info('X-y shapes: %s, %s' % (X.shape, y.shape))
    logging.info('True fraction: %s' % y.mean())

    # How large is the dataset? I might consider writing it out eventually
    logging.info('Size of X: %g MB' % (X.dtype.itemsize * X.size / 1e6))
    logging.info('Size of y: %g MB' % (y.dtype.itemsize * y.size / 1e6))

    # For now, use full input sample as training sample
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    y_train = y

    # Train classifiers
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    logging.info('Training LogisticRegression')
    lr_clf = LogisticRegression()
    lr_clf.fit(X_train, y_train)

    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    logging.info('Training a RandomForestClassifier')
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)

    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    logging.info('Training an AdaBoostClassifier')
    abdt_clf = AdaBoostClassifier()
    abdt_clf.fit(X_train, y_train)

    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    logging.info('Training a GradientBoostingClassifier')
    gbdt_clf = GradientBoostingClassifier()
    gbdt_clf.fit(X_train, y_train)

    # http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    logging.info('Training an MLPClassifier')
    mlp_clf = MLPClassifier()
    mlp_clf.fit(X_train, y_train)

    # Save the classifiers
    # http://scikit-learn.org/stable/modules/model_persistence.html
    with open(args.output_file, 'wb') as f:
        pickle.dump({
            'lr': lr_clf,
            'rf': rf_clf,
            'abdt': abdt_clf,
            'gbdt': gbdt_clf, 
            'mlp': mlp_clf, 
            'scaler': scaler},
            f)

    # Other stuff
    # http://scikit-learn.org/stable/modules/cross_validation.html
    # http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html

    logging.info('All done!')

if __name__ == '__main__':
    main()

#!/usr/bin/env python

"""
This python script applies scikit-learn classifiers to the ATLAS RPV data
"""

import os
import argparse
import logging

import multiprocessing as mp

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.metrics
import matplotlib.pyplot as plt

def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('run_classifiers')
    add_arg = parser.add_argument
    add_arg('input_dir', help='Directory of npz files')
    add_arg('-p', '--num-workers', type=int, default=1,
            help='Number of concurrent worker processes')
    add_arg('--sig', default='rpv_1400_850',
            help='Signal sample to use')
    add_arg('--bkg', nargs='*',
            default=['qcd_JZ3', 'qcd_JZ4', 'qcd_JZ5', 'qcd_JZ6',
                     'qcd_JZ7', 'qcd_JZ8', 'qcd_JZ9', 'qcd_JZ10',
                     'qcd_JZ11', 'qcd_JZ12', 'rpv_1400_850'],
            help='Background sample names to use')
    add_arg('--num-sig', type=int, help='Number of signal events to use')
    add_arg('--num-bkg', type=int,
            help='Number of bkg events to use (per sample)')
    add_arg('-w', '--apply-weights', action='store_true',
            help='Use event weights for classifier evaluation')
    add_arg('-i', '--interactive', action='store_true',
            help='Embed IPython session at end of processing for interactive work')
    return parser.parse_args()

def get_file_keys(file_name):
    """Retrieves the list of keys from an npz file"""
    with np.load(file_name) as f:
        keys = f.keys()
    return keys

def retrieve_data(file_name, *keys):
    """
    A helper function for retrieving some specified arrays from one npz file.
    Returns a list of arrays corresponding to the requested key name list.
    """
    with np.load(file_name) as f:
        try:
            data = [f[key] for key in keys]
        except KeyError as err:
            logging.error('Requested key not found. Available keys: %s' % f.keys())
            raise
    return data

def parse_object_features(array, num_objects, default_val=0.):
    """
    Takes an array of object arrays and returns a fixed 2D array.
    Clips and pads each element as necessary.
    Output shape is (array.shape[0], num_objects).
    """
    # Create the output first
    length = array.shape[0]
    output_array = np.full((length, num_objects), default_val)
    # Fill the output
    for i in xrange(length):
        k = min(num_objects, array[i].size)
        output_array[i,:k] = array[i][:k]
    return output_array

def prepare_sample_features(sample_file, num_jets=4, max_events=None):
    """Load the model features from a sample file"""
    data = retrieve_data(
        sample_file, 'fatJetPt', 'fatJetEta', 'fatJetPhi', 'fatJetM')
    num_events = data[0].shape[0]
    if max_events is not None and max_events < num_events:
        data = [d[:max_events] for d in data]
    return np.hstack(parse_object_features(a, num_jets) for a in data)

def get_sample_weight(sample_file, lumi=1000.):
    """Calculate the weight for one sample"""
    xsec, tot_events = retrieve_data(sample_file, 'xsec', 'totalEvents')
    assert np.unique(xsec).size == 1
    return xsec[0] * lumi / tot_events.sum()

def calc_fpr_tpr(y_true, y_pred, w):
    """Calculate false-positive and true-positive rates"""
    tp = (y_true * y_pred * w).sum()
    fp = (np.logical_not(y_true) * y_pred * w).sum()
    nsig = (y_true * w).sum()
    nbkg = w.sum() - nsig
    tpr = tp / nsig
    fpr = fp / nbkg
    return fpr, tpr

def report_classifier(classifier, X_train, X_test, y_train, y_test,
                      w_train=None, w_test=None):
    pred_test = classifier.predict(X_test)
    acc_train = classifier.score(X_train, y_train) # sample_weight=w_train)
    acc_test = classifier.score(X_test, y_test) # sample_weight=w_test)
    class_report = sklearn.metrics.classification_report(
        y_test, pred_test, sample_weight=w_test, target_names=['QCD', 'RPV'])
    logging.info('Unweighted train set accuracy: %f' % acc_train)
    logging.info('Unweighted test set accuracy: %f' % acc_test)
    logging.info('Unweighted test set classification report:\n%s' % class_report)

def calc_classifier_metrics(classifiers, X_test, y_test, w_test=None):
    """Calculates false-positive and true-positive rates and AUC
    for each classifier and returns the arrays"""
    # Calculate the test set class probability scores
    probs = [clf.predict_proba(X_test)[:,1] for clf in classifiers]
    # Calculate the metrics
    rates = [sklearn.metrics.roc_curve(y_test, p, sample_weight=w_test)
             for p in probs]
    # Reduce precision to fix rounding issues
    fpr = [r[0].astype(np.float32) for r in rates]
    tpr = [r[1].astype(np.float32) for r in rates]
    auc = [sklearn.metrics.auc(f, t) for (f, t) in zip(fpr, tpr)]
    return fpr, tpr, auc

def main():
    """Main execution function"""

    # Parse command line
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Initializing')

    logging.info('Configuration: %s' % args)

    sig_sample = args.sig
    bkg_samples = args.bkg
    path_prep = lambda s: os.path.join(args.input_dir, s + '.npz')
    sig_file = path_prep(sig_sample)
    bkg_files = map(path_prep, bkg_samples)

    logging.info('Sig file: %s' % sig_file)
    logging.info('Bkg files: %s' % bkg_files)

    logging.info('Preparing features')
    sample_files = [sig_file] + bkg_files
    bkg_features = [prepare_sample_features(f, max_events=args.num_bkg)
                    for f in bkg_files]
    sig_features = prepare_sample_features(sig_file, max_events=args.num_sig)
    sample_features = [sig_features] + bkg_features
    sample_events = [sf.shape[0] for sf in sample_features]
    sample_labels = [1.] + [0.]*len(bkg_files)
    # Retrieve the analysis SR flag
    sample_passSR = [retrieve_data(f, 'passSR')[0][:nevt]
                     for (f, nevt) in zip(sample_files, sample_events)]
    # Sample weights
    if args.apply_weights:
        sample_weights = [get_sample_weight(f) for f in sample_files]
    else:
        sample_weights = [1. for f in sample_files]

    # Merge the feature vectors
    X = np.concatenate(sample_features)
    y = np.concatenate([np.full(nevt, l) for (nevt, l) in
                        zip(sample_events, sample_labels)])
    passSR = np.concatenate(sample_passSR)
    weights = np.concatenate([np.full(nevt, w) for (nevt, w) in
                              zip(sample_events, sample_weights)])

    logging.info('X-y shapes: %s, %s' % (X.shape, y.shape))
    logging.info('True fraction: %s' % y.mean())

    # How large is the dataset? I might consider writing it out eventually
    logging.info('Size of X: %g MB' % (X.dtype.itemsize * X.size / 1e6))
    logging.info('Size of y: %g MB' % (y.dtype.itemsize * y.size / 1e6))

    # Split into training and test samples
    (X_train, X_test, y_train, y_test,
        w_train, w_test, passSR_train, passSR_test) = (
            train_test_split(X, y, weights, passSR))

    # Evaluate the existing analysis cuts
    logging.info('Classification report for SR:\n%s' %
        sklearn.metrics.classification_report(
            y_test, passSR_test, target_names=['Background', 'Signal']))

    # Calculate TPR and FPR for passSR
    sr_fpr, sr_tpr = calc_fpr_tpr(y_test, passSR_test, w_test)
    logging.info('SR FPR: %f, TPR: %f' % (sr_fpr, sr_tpr))

    logging.info('Training LogisticRegression')
    lr_clf = make_pipeline(StandardScaler(), LogisticRegression())
    lr_clf.fit(X_train, y_train)
    report_classifier(lr_clf, X_train, X_test, y_train, y_test, w_train, w_test)

    logging.info('Training a DecisionTreeClassifier')
    dt_clf = make_pipeline(StandardScaler(), DecisionTreeClassifier())
    dt_clf.fit(X_train, y_train)
    report_classifier(dt_clf, X_train, X_test, y_train, y_test, w_train, w_test)

    logging.info('Training a RandomForestClassifier')
    rf_clf = make_pipeline(StandardScaler(), RandomForestClassifier())
    rf_clf.fit(X_train, y_train)
    report_classifier(rf_clf, X_train, X_test, y_train, y_test, w_train, w_test)

    logging.info('Training a GradientBoostingClassifier')
    bdt_clf = make_pipeline(StandardScaler(), GradientBoostingClassifier())
    bdt_clf.fit(X_train, y_train)
    report_classifier(bdt_clf, X_train, X_test, y_train, y_test, w_train, w_test)

    logging.info('Training an MLPClassifier')
    mlp_clf = make_pipeline(StandardScaler(), MLPClassifier())
    mlp_clf.fit(X_train, y_train)
    report_classifier(mlp_clf, X_train, X_test, y_train, y_test, w_train, w_test)

    classifiers = [lr_clf, rf_clf, bdt_clf, mlp_clf]
    clf_names = ['LR', 'RF', 'BDT', 'MLP']
    clf_fpr, clf_tpr, clf_auc = calc_classifier_metrics(classifiers, X_test, y_test, w_test)

    # Save the output arrays for later plotting, etc.
    np.savez('sklearn_roc.npz', fpr=clf_fpr, tpr=clf_tpr, auc=clf_auc, names=clf_names)

    # Plot the ROC curves
    rocFig = plt.figure()
    # Plot the SR point
    plt.plot(sr_fpr, sr_tpr, 's', label='Ana SR')
    # Plot the classifiers
    for i in range(len(classifiers)):
        label = clf_names[i] + ', AUC=%.3f' % clf_auc[i]
        plt.plot(clf_fpr[i], clf_tpr[i], label=label)
    plt.legend(loc=0)
    plt.xlim((0, 0.001))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    # Save the figure
    rocFig.savefig('sklearn_roc.png')

    logging.info('All done!')

    # Open an interactive session if requested
    if args.interactive:
        import IPython
        IPython.embed()

if __name__ == '__main__':
    main()

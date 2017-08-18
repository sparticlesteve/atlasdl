#!/usr/bin/env python

"""
This python script takes the classifiers trained via the train_classifiers
script and evaluates them on a fixed test set of features.
"""

# System imports
import argparse
import logging
import pickle

# External imports
import numpy as np
import sklearn.metrics

def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('eval_classifiers')
    add_arg = parser.add_argument
    add_arg('input_file', help='Input npz file of features to evaluate on')
    add_arg('classifiers_file', help='Input pkl file of trained classifiers')
    add_arg('-r', '--roc-file', help='Output npz file of classifier ROC curves')
    add_arg('-i', '--interactive', action='store_true',
            help='Drop into IPython at end')
    return parser.parse_args()

def calc_fpr_tpr(y_true, y_pred, w):
    """Calculate false-positive and true-positive rates"""
    tp = (y_true * y_pred * w).sum()
    fp = (np.logical_not(y_true) * y_pred * w).sum()
    nsig = (y_true * w).sum()
    nbkg = w.sum() - nsig
    tpr = tp / nsig
    fpr = fp / nbkg
    return fpr, tpr

def report_classifier(classifier, X, y, w=None):
    pred = classifier.predict(X)
    acc = classifier.score(X, y) # sample_weight=w)
    class_report = sklearn.metrics.classification_report(
        y, pred, sample_weight=w, target_names=['QCD', 'RPV'])
    logging.info('Unweighted test set accuracy: %f' % acc)
    logging.info('Unweighted test set classification report:\n%s' % class_report)

def calc_classifier_metrics(classifiers, X, y, w=None):
    """Calculates false-positive and true-positive rates and AUC
    for each classifier and returns the arrays"""
    # Calculate the test set class probability scores
    probs = [clf.predict_proba(X)[:,1] for clf in classifiers]
    # Calculate the metrics
    rates = [sklearn.metrics.roc_curve(y, p, sample_weight=w)
             for p in probs]
    # Reduce precision to fix rounding issues
    fpr = [r[0].astype(np.float32) for r in rates]
    tpr = [r[1].astype(np.float32) for r in rates]
    auc = [sklearn.metrics.auc(f, t) for (f, t) in zip(fpr, tpr)]
    return fpr, tpr, auc

def main():
    """Main execution function"""
    # Command line parsing
    args = parse_args()
    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Initializing')
    logging.info('Configuration: %s' % args)

    # Load the input classifiers
    with open(args.classifiers_file) as f:
        data = pickle.load(f)
        scaler = data['scaler']
        classifiers = dict((key, val) for (key, val) in data.items()
                           if key != 'scaler')

    # Load the input feature data
    with np.load(args.input_file) as f:
        X = f['X']
        y = f['y']
        passSR = f['passSR']
        weights = f['weights']

    # Apply the standard scaler
    X = scaler.transform(X)

    # Evaluate the existing analysis cuts
    logging.info('Classification report for SR:\n%s' %
        sklearn.metrics.classification_report(
            y, passSR, target_names=['Background', 'Signal']))

    # Calculate TPR and FPR for passSR
    sr_fpr, sr_tpr = calc_fpr_tpr(y, passSR, weights)
    logging.info('SR FPR: %f, TPR: %f' % (sr_fpr, sr_tpr))

    for clf_name, clf in classifiers.items():
        logging.info('Evaluating classifier: %s' % clf_name)
        report_classifier(clf, X, y, weights)

    clf_fpr, clf_tpr, clf_auc = calc_classifier_metrics(classifiers.values(), X, y, weights)
    #clf_names = ['LR', 'RF', 'BDT', 'MLP']

    # Save the output arrays for later plotting, etc.
    if args.roc_file is not None:
        np.savez(args.roc_file, fpr=clf_fpr, tpr=clf_tpr, auc=clf_auc, names=classifiers.keys())

    if args.interactive:
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()

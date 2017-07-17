#!/usr/bin/env python

"""
This python script runs on the pre-processed ATLAS RPV data in
npz format and prepares the feature set for classifier training input.
"""

# System imports
import os
import argparse
import logging

# External imports
import numpy as np

# Local imports
from physics_selections import sum_fatjet_mass, fatjet_deta12

def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('extract_features')
    add_arg = parser.add_argument
    add_arg('input_dir', help='Directory of npz files')
    add_arg('output_file', help='Name of output npz file')
    add_arg('--sig', default='rpv_1400_850',
            help='Signal sample to use')
    add_arg('--bkg', nargs='*',
            default=['qcd_JZ3', 'qcd_JZ4', 'qcd_JZ5', 'qcd_JZ6',
                     'qcd_JZ7', 'qcd_JZ8', 'qcd_JZ9', 'qcd_JZ10',
                     'qcd_JZ11', 'qcd_JZ12'],
            help='Background sample names to use')
    add_arg('--num-sig', type=int, help='Number of signal events to use')
    add_arg('--num-bkg', type=int,
            help='Number of bkg events to use (per sample)')
    return parser.parse_args()

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

def prepare_sample_features(sample_file, max_jets=5, max_events=None):
    """Load the model features from a sample file"""
    print(sample_file)
    # Pull the variables from the input file
    data = retrieve_data(sample_file, 'fatJetPt', 'fatJetEta',
                         'fatJetPhi', 'fatJetM')
    num_events = data[0].shape[0]
    if max_events is not None and max_events < num_events:
        data = [d[:max_events] for d in data]

    # Calculate event-level features like numJets, sum jet mass, etc.
    jetPt, jetEta, jetPhi, jetM = data
    numJet = np.vectorize(lambda x: x.shape[0])(jetPt)
    sumMass = np.vectorize(sum_fatjet_mass)(jetM)
    jetDEta12 = np.vectorize(fatjet_deta12)(jetEta)
    evt_features = [numJet[:,None], sumMass[:,None], jetDEta12[:,None]]

    # Parse out the object features
    obj_features = [parse_object_features(a, max_jets) for a in data]

    return np.hstack(evt_features + obj_features)

def get_sample_weight(sample_file, lumi=1000.):
    """Calculate the weight for one sample"""
    xsec, tot_events = retrieve_data(sample_file, 'xsec', 'totalEvents')
    assert np.unique(xsec).size == 1
    return xsec[0] * lumi / tot_events.sum()

def main():
    """Main execution function"""

    # Parse command line
    args = parse_args()

    # Setup
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
    sample_weights = [get_sample_weight(f) for f in sample_files]

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

    # Write the output features
    np.savez_compressed(args.output_file, X=X, y=y,
                        passSR=passSR, weights=weights)
    #import IPython; IPython.embed()

if __name__ == '__main__':
    main()

#!/usr/bin/env python

"""
This script will do pre-processing of input data
"""

import os
import argparse
import logging

import multiprocessing as mp

import numpy as np

from physics_selections import (filter_objects, filter_events,
                                select_fatjets, is_baseline_event,
                                sum_fatjet_mass, fatjet_deta12,
                                pass_sr4j, pass_sr5j,
                                is_signal_region_event)
#from weights import (get_xaod_rpv_params, get_xaod_bkg_xsec, get_xaod_sumw,
#                     get_delphes_xsec, get_delphes_sumw)
from utils import suppress_stdout_stderr


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser('prepare_data')
    add_arg = parser.add_argument
    add_arg('input_file_list', nargs='+',
            help='Text file of input files')
    #add_arg('--input-type', default='xaod', choices=['xaod', 'delphes'],
    #        help='Specify xaod or delphes input file type')
    add_arg('-o', '--output-npz', help='Output numpy binary file')
    #add_arg('--output-h5', help='Output hdf5 file')
    add_arg('-n', '--max-events', type=int,
            help='Maximum number of events to read')
    add_arg('-p', '--num-workers', type=int, default=0,
            help='Number of concurrent worker processes')
    #add_arg('--write-clus', action='store_true',
    #        help='Write cluster info to output')
    #add_arg('--write-fjets', action='store_true',
    #        help='Write fat jet info to output')
    #add_arg('--write-mass', action='store_true',
    #        help='Write RPV theory mass params to output')
    #add_arg('--write-tracks', action='store_true',
    #        help='Write ID track variables')
    #add_arg('--bins', default=64, type=int,
    #        help='The number of bins aka the dimensions of the hist data')
    return parser.parse_args()

def get_data(files, branch_dict, **kwargs):
    """Applies root_numpy to get out a numpy array"""
    import root_numpy as rnp
    try:
        with suppress_stdout_stderr():
            tree = rnp.root2array(files, branches=branch_dict.keys(),
                                  warn_missing_tree=True, **kwargs)
    except IOError as e:
        logging.warn('WARNING: root2array gave an IOError:', e)
        return None
    # Convert immutable structured array into dictionary of arrays
    data = dict()
    for (oldkey, newkey) in branch_dict.items():
        data[newkey] = tree[oldkey]
    return data

def process_events(data):
    """Applies physics selections and filtering"""

    # Object selection
    vec_select_fatjets = np.vectorize(select_fatjets, otypes=[np.ndarray])
    fatJetPt, fatJetEta = data['fatJetPt'], data['fatJetEta']
    jetIdx = vec_select_fatjets(fatJetPt, fatJetEta)

    fatJetPt, fatJetEta, fatJetPhi, fatJetM = filter_objects(
        jetIdx, fatJetPt, fatJetEta, data['fatJetPhi'], data['fatJetM'])

    # Baseline event selection
    skimIdx = np.vectorize(is_baseline_event)(fatJetPt)
    fatJetPt, fatJetEta, fatJetPhi, fatJetM = filter_events(
        skimIdx, fatJetPt, fatJetEta, fatJetPhi, fatJetM)
    num_baseline = np.sum(skimIdx)
    num_total = len(skimIdx)
    logging.info('Baseline selected events: %d / %d' % (num_baseline, num_total))

    # Calculate quantities needed for SR selection
    if num_baseline > 0:
        numFatJet = np.vectorize(lambda x: x.size)(fatJetPt)
        sumFatJetM = np.vectorize(sum_fatjet_mass)(fatJetM)
        fatJetDEta12 = np.vectorize(fatjet_deta12)(fatJetEta)

        # Signal-region event selection
        passSR4J = np.vectorize(pass_sr4j)(numFatJet, sumFatJetM, fatJetDEta12)
        passSR5J = np.vectorize(pass_sr5j)(numFatJet, sumFatJetM, fatJetDEta12)
        passSR = np.logical_or(passSR4J, passSR5J)
    else:
        numFatJet = sumFatJetM = fatJetDEta12 = np.zeros(0)
        passSR4J = passSR5J = passSR = np.zeros(0, dtype=np.bool)

    # Prepare the skimmed results
    skimData =  dict(fatJetPt=fatJetPt, fatJetEta=fatJetEta,
                     fatJetPhi=fatJetPhi, fatJetM=fatJetM,
                     sumFatJetM=sumFatJetM, passSR4J=passSR4J,
                     passSR5J=passSR5J, passSR=passSR)

    # Get the remaining unskimmed columns, and skim them
    keys = set(data.keys()) - set(skimData.keys())
    for k in keys:
        skimData[k] = data[k][skimIdx]

    # Finally, add some bookkeeping data
    skimData['totalEvents'] = np.array([num_total])
    skimData['skimEvents'] = np.array([num_baseline])

    return skimData

def filter_delphes_to_numpy(files, max_events=None):
    """Processes some files by converting to numpy and applying filtering"""

    if type(files) != list:
        files = [files]

    # Branch name remapping for convenience
    branch_dict = {
        'Event.Number' : 'eventNumber',
        'Event.ProcessID' : 'proc',
        'Tower.Eta' : 'clusEta',
        'Tower.Phi' : 'clusPhi',
        'Tower.E' : 'clusE',
        'Tower.Eem' : 'clusEM',
        'FatJet.PT' : 'fatJetPt',
        'FatJet.Eta' : 'fatJetEta',
        'FatJet.Phi' : 'fatJetPhi',
        'FatJet.Mass' : 'fatJetM',
        'Track.PT' : 'trackPt',
        'Track.Eta' : 'trackEta',
        'Track.Phi' : 'trackPhi',
    }

    # Convert the data to numpy
    logging.info('Now processing: %s' % files)
    data = get_data(files, branch_dict, treename='Delphes', stop=max_events)
    if data is None:
        return None

    # Apply physics
    logging.info('Applying event selection')
    skimData = process_events(data)

    # Get the sample config string from the filenames.
    # For now, out of laziness, allow only one sample at a time.
    # NOTE: this assumes a particular naming convention of the delphes files!!
    #samples = map(lambda s: os.path.basename(s).split('-')[0], files)
    #if np.unique(samples).size > 1:
    #    raise Exception('Mixing delphes samples not yet supported: ' + str(samples))

    # Store the sample name for metadata lookups
    #num_event = results['tree'].shape[0]
    #results['sample'] = np.full(num_event, samples[0], 'S30')

    return skimData

def get_calo_image(tree, xkey='clusEta', ykey='clusPhi', wkey='clusE',
                   bins=100, xlim=[-2.5, 2.5], ylim=[-3.15, 3.15]):
    """Convert the numpy structure with calo clusters into 2D calo histograms"""
    # Bin the data and reshape so we can concatenate along first axis into a 3D array.
    def hist2d(x, y, w):
        return (np.histogram2d(x, y, bins=bins, weights=w, range=[xlim, ylim])[0]
                .reshape([1, bins, bins]))
    hist_list = map(hist2d, tree[xkey], tree[ykey], tree[wkey])
    return np.concatenate(hist_list)

def get_track_image(tree, xkey='trackEta', ykey='trackPhi',
                   bins=100, xlim=[-2.5, 2.5], ylim=[-3.15, 3.15]):
    """Convert the numpy structure with calo clusters into 2D calo histograms"""
    # Bin the data and reshape so we can concatenate along first axis into a 3D array.
    def hist2d(x, y):
        return (np.histogram2d(x, y, bins=bins, range=[xlim, ylim])[0]
                .reshape([1, bins, bins]))
    hist_list = map(hist2d, tree[xkey], tree[ykey])
    return np.concatenate(hist_list)

def merge_results(dicts):
    """Merge a list of dictionaries with numpy arrays"""
    dicts = filter(None, dicts)
    # First, get the list of unique keys
    keys = set(key for d in dicts for key in d.keys())
    result = dict()
    for key in keys:
        arrays = [d[key] for d in dicts]
        result[key] = np.concatenate([d[key] for d in dicts])
    return result

def get_meta_data_delphes(sample_names):
    if sample_names is None:
        logging.warn('no sample_names => no metadata => no event weights')
        return None, None, None, None
    # TODO: parse these out from the sample name
    mglu, mneu = None, None
    xsec = np.vectorize(get_delphes_xsec)(sample_names)
    sumw = np.vectorize(get_delphes_sumw)(sample_names)
    return mglu, mneu, xsec, sumw

def get_event_weights(xsec, mcw, sumw, lumi=36000):
    """Calculate event weights"""
    # Need to extract the first entry of the generator weights per event
    if type(mcw) == np.ndarray:
        mcw = np.vectorize(lambda g: g[0])(mcw)
    return xsec * mcw * lumi / sumw

def main():
    """Main execution function"""
    args = parse_args()

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Initializing')

    # Get the input file list
    input_files = []
    for input_list in args.input_file_list:
        with open(input_list) as f:
            input_files.extend(map(str.rstrip, f.readlines()))
    logging.info('Processing %i input files' % len(input_files))

    filter_func = filter_delphes_to_numpy

    # Parallel processing
    if args.num_workers > 0:
        logging.info('Starting process pool of %d workers' % args.num_workers)
        # Create a pool of workers
        pool = mp.Pool(processes=args.num_workers)
        # Convert to numpy structure in parallel
        task_data = pool.map(filter_func, input_files)
        # Merge the results from each task
        data = merge_results(task_data)
    # Sequential processing
    else:
        # Run the conversion and filter directly
        data = filter_func(input_files, args.max_events)

    # Abort gracefully if no events survived skimming
    if data['skimEvents'].sum() == 0:
        logging.info('No events selected by filter. Exiting.')
        return

    # Write results to npz file
    if args.output_npz is not None:
        logging.info('Writing output to %s' % args.output_npz)
        np.savez_compressed(args.output_npz, **data)

    # TODO: finish picking up stuff from below

    # Get the 2D histogram
    #data['hist'] = get_calo_image(tree, bins=args.bins)
    #data['histEM'] = get_calo_image(tree, xkey='clusEta', ykey='clusPhi', wkey='clusEM', bins=args.bins)
    #data['histtrack'] = get_track_image(tree, bins=args.bins)

    # Get sample metadata
    #if args.input_type == 'xaod':
    #    mglu, mneu, xsec, sumw = get_meta_data_xaod(tree['dsid'])
    #    mcw = tree['genWeight']
    #else:
    #    mglu, mneu, xsec, sumw = get_meta_data_delphes(data.get('sample', None))
    #    mcw = 1

    # Calculate the event weights
    #data['weight'] = (get_event_weights(xsec, mcw, sumw)
    #                  if sumw is not None else None)

    # Signal region flags
    #passSR4J = data['passSR4J']
    #passSR5J = data['passSR5J']
    #passSR = data['passSR']

    # Dictionary of output data
    #outputs = {}
    #output_keys = ['hist', 'histEM', 'histtrack', 'passSR4J', 'passSR5J', 'passSR', 'weight']

    # Addition optional outputs
    #if args.write_clus:
    #    for key in ['clusEta', 'clusPhi', 'clusE', 'clusEM']:
    #        try:
    #            outputs[key] = tree[key]
    #        except KeyError:
    #            logging.warn('Failed to write missing key:', key)
    #if args.write_fjets:
    #    output_keys += ['fatJetPt', 'fatJetEta', 'fatJetPhi', 'fatJetM']
    #if args.write_mass:
    #    if mglu is not None:
    #        outputs['mGlu'] = mglu
    #    if mneu is not None:
    #        outputs['mNeu'] = mneu
    #if args.write_tracks:
    #    output_keys += ['trackEta', 'trackPhi']

    #for key in output_keys:
    #    try:
    #        outputs[key] = data[key]
    #    except KeyError:
    #        logging.warn('Failed to write missing key:', key)
    #        raise

    # Print some summary information
    #logging.info('SR4J selected events: %d / %d' % (np.sum(passSR4J), tree.size))
    #weight = data['weight']
    #if weight is not None:
    #    logging.info('SR4J weighted events: %f' % np.sum(weight[passSR4J]))
    #logging.info('SR5J selected events: %d / %d' % (np.sum(passSR5J), tree.size))
    #if weight is not None:
    #    logging.info('SR5J weighted events: %f' % np.sum(weight[passSR5J]))
    #logging.info('SR selected events: %d / %d' % (np.sum(passSR), tree.size))
    #if weight is not None:
    #    logging.info('SR weighted events: %f' % (np.sum(weight[passSR])))

    # Write results to hdf5
    #if args.output_h5 is not None:
    #    logging.info('Writing output to', args.output_h5)
    #    write_hdf5(args.output_h5, outputs)

    logging.info('Done!')

if __name__ == '__main__':
    main()

#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py -d labels -o outputs -s scores.csv
#
# where 'labels' is a folder containing files with the labels, 'outputs' is a folder containing files with the outputs from your
# model, and 'scores.csv' (optional) is a collection of scores for the model outputs.
#
# Each label or output file must have the format described on the Challenge webpage. The scores for the algorithm outputs are also
# described on the Challenge webpage.

import argparse
import numpy as np
import os
import os.path
import sys


from datetime import datetime

from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Evaluate the Challenge model(s).'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--label_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-x', '--extra_scores', action='store_true')
    parser.add_argument('-s', '--score_file', type=str, required=False)
    return parser

# Evaluate the models.
def evaluate_model(label_folder, output_folder, extra_scores=False):
    # Find the records.
    records = find_records(label_folder)
    num_records = len(records)

    # # Compute the signal reconstruction metrics.
    # records_completed_signal_reconstruction = list()
    # snr = dict()
    # snr_median = dict()
    # ks_metric = dict()
    # asci_metric = dict()
    # weighted_absolute_difference_metric = dict()

    # # Iterate over the records.
    # for record in records:
    #     # Load the signals, if available.
    #     label_record = os.path.join(label_folder, record)
    #     label_signal, label_fields = load_signal(label_record)

    #     if label_signal is not None:
    #         label_channels = label_fields['sig_name']
    #         label_num_channels = label_fields['n_sig']
    #         label_num_samples = label_fields['sig_len']
    #         label_sampling_frequency = label_fields['fs']
    #         label_units = label_fields['units']

    #         output_record = os.path.join(output_folder, record)
    #         output_signal, output_fields = load_signal(output_record)

    #         if output_signal is not None:
    #             output_channels = output_fields['sig_name']
    #             output_num_channels = output_fields['n_sig']
    #             output_num_samples = output_fields['sig_len']
    #             output_sampling_frequency = output_fields['fs']
    #             output_units = output_fields['units']

    #             records_completed_signal_reconstruction.append(record)

    #             # Check that the label and output signals match as expected.
    #             assert(label_sampling_frequency == output_sampling_frequency)
    #             assert(label_units == output_units)

    #             # Reorder the channels in the output signal to match the channels in the label signal.
    #             output_signal = reorder_signal(output_signal, output_channels, label_channels)

    #             # Trim or pad the channels in the output signal to match the channels in the label signal.
    #             output_signal = trim_signal(output_signal, label_num_samples)

    #             # Replace the samples with NaN values in the output signal with zeros.
    #             output_signal[np.isnan(output_signal)] = 0

    #         else:
    #             output_signal = np.zeros(np.shape(label_signal), dtype=label_signal.dtype)

    #         # Compute the signal reconstruction metrics.
    #         channels = label_channels
    #         num_channels = label_num_channels
    #         sampling_frequency = label_sampling_frequency

    #         for j, channel in enumerate(channels):
    #             value = compute_snr(label_signal[:, j], output_signal[:, j])
    #             snr[(record, channel)] = value

    #             if extra_scores:
    #                 value = compute_snr_median(label_signal[:, j], output_signal[:, j])
    #                 snr_median[(record, channel)] = value

    #                 value = compute_ks_metric(label_signal[:, j], output_signal[:, j])
    #                 ks_metric[(record, channel)] = value

    #                 value = compute_asci_metric(label_signal[:, j], output_signal[:, j])
    #                 asci_metric[(record, channel)] = value

    #                 value = compute_weighted_absolute_difference(label_signal[:, j], output_signal[:, j], sampling_frequency)
    #                 weighted_absolute_difference_metric[(record, channel)] = value

    # # Compute the metrics.
    # if len(records_completed_signal_reconstruction) > 0:
    #     snr = np.array(list(snr.values()))
    #     if not np.all(np.isnan(snr)):
    #         mean_snr = np.nanmean(snr)
    #     else:
    #         mean_snr = float('nan')

    #     if extra_scores:
    #         snr_median = np.array(list(snr_median.values()))
    #         if not np.all(np.isnan(snr_median)):
    #             mean_snr_median = np.nanmean(snr_median)
    #         else:
    #             mean_snr_median = float('nan')

    #         ks_metric = np.array(list(ks_metric.values()))
    #         if not np.all(np.isnan(ks_metric)):
    #             mean_ks_metric = np.nanmean(ks_metric)
    #         else:
    #             mean_ks_metric = float('nan')

    #         asci_metric = np.array(list(asci_metric.values()))
    #         if not np.all(np.isnan(asci_metric)):
    #             mean_asci_metric = np.nanmean(asci_metric)
    #         else:
    #             mean_asci_metric = float('nan')

    #         weighted_absolute_difference_metric = np.array(list(weighted_absolute_difference_metric.values()))
    #         if not np.all(np.isnan(weighted_absolute_difference_metric)):
    #             mean_weighted_absolute_difference_metric = np.nanmean(weighted_absolute_difference_metric)
    #         else:
    #             mean_weighted_absolute_difference_metric = float('nan')
    #     else:
    #         mean_snr_median = float('nan')
    #         mean_ks_metric = float('nan')
    #         mean_asci_metric = float('nan')
    #         mean_weighted_absolute_difference_metric = float('nan')

    # else:
    #     mean_snr = float('nan')
    #     mean_snr_median = float('nan')
    #     mean_ks_metric = float('nan')
    #     mean_asci_metric = float('nan')
    #     mean_weighted_absolute_difference_metric = float('nan')

    # Compute the classification metrics.
    records_completed_classification = list()
    label_dxs = list()
    output_dxs = list()

    # Iterate over the records.
    for record in records:
        # Load the classes, if available.
        label_record = os.path.join(label_folder, record)
        label_dx = load_dx(label_record)

        if label_dx:
            output_record = os.path.join(output_folder, record)
            output_dx = load_dx(output_record)

            if output_dx:
                records_completed_classification.append(record)

            label_dxs.append(label_dx)
            output_dxs.append(output_dx)
    
    num_normal_predict = 0
    num_abnormal_predict = 0
    for ele in output_dxs:
        if ele[1] == 'Normal':
            num_normal_predict +=1
        else:
            num_abnormal_predict +=1

    moretex = f"{label_dxs.count(['Normal'])} Normal in data \n {num_normal_predict} Normal in predict \n {label_dxs.count(['Abnormal'])} Abnormal in data \n {num_abnormal_predict} Abnormal in predict\n"


    # Compute the metrics.
    if len(records_completed_classification) > 0:
        f_measure, perclass_f_measure, _ = compute_f_measure(label_dxs, output_dxs)
        mean_accuracy, per_class_accuracy, classes = compute_accuracy(label_dxs, output_dxs)
        print("Per class F measure:", perclass_f_measure)
        print("F-measure: ", f_measure)
        print("Per class accuracy", per_class_accuracy)
        print("Mean accuracy:", mean_accuracy)
    else:
        f_measure = float('nan')

    # Get the current datetime to report
    current_time = datetime.now()

    # Format the datetime into a string with day, month, hour, and minute
    formatted_time = current_time.strftime("%H:%M, %d, %B")


    # Return the results.
    #return mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, mean_weighted_absolute_difference_metric, f_measure
    output_string = f"\n---------------------------------------------------------------------------------------------------------------------------------------------------------\n Label folder: {label_folder}\n Output folder: {output_folder} \n {formatted_time} \n Classes: {classes} \n\n {moretex} \n Per_class_accuracy: {per_class_accuracy} \n Mean accuracy: {mean_accuracy:.3f} \n\n Perclass_f_measure: {perclass_f_measure} \n F-measure: {f_measure:.3f}"
    return output_string

#NHATedit
# Run the code.
def run(args):
    # Compute the scores for the model outputs.
    output_string = evaluate_model(args.label_folder, args.output_folder, args.extra_scores)
    output_string = output_string 
    if args.score_file:
        with open(args.score_file, 'a') as file:
            file.write(output_string)
    else:
        print(output_string)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
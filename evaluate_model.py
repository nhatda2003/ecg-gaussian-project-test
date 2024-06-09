

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
            
    #Some text to put in scores.csv
    print(label_dxs.count(['Normal']))
    print(label_dxs.count(['Abnormal']))
    moretex = f"{label_dxs.count(['Normal'])} Normal in data \n {output_dxs.count(['Normal'])} Normal in predict \n {label_dxs.count(['Abnormal'])} Abnormal in data \n {output_dxs.count(['Abnormal'])}  Abnormal in predict\n Ratio in data:      {int(round(label_dxs.count(['Normal'])/len(label_dxs),2)*100)} Normal : {int(round(label_dxs.count(['Abnormal'])/len(label_dxs),2)*100)} Abnormal\n"


    # Compute the metrics.
    if len(records_completed_classification) > 0:
        f_measure, perclass_f_measure, _ = compute_f_measure(label_dxs, output_dxs)
        all_accuracy, per_class_accuracy, classes = compute_accuracy(label_dxs, output_dxs)
        print("Per class F measure:", perclass_f_measure)
        print("F-measure: ", f_measure)
        print("Per class accuracy", per_class_accuracy)
        print("All accuracy:", all_accuracy)
    else:
        f_measure = float('nan')

    # Get final text to put in scores.csv for report save
    current_time = datetime.now()
    formatted_time = current_time.strftime("%H:%M, %d, %B")
    output_string = f"\n---------------------------------------------------------------------------------------------------------------------------------------------------------\n Label folder: {label_folder}\n Output folder: {output_folder} \n {formatted_time} \n Classes: {classes} \n\n {moretex} \n Per_class_accuracy [Normal  Abnormal]: {per_class_accuracy} \n All accuracy: {all_accuracy:.3f} \n\n Perclass_f_measure [Normal  Abnormal]: {perclass_f_measure} \n F-measure: {f_measure:.3f}"
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